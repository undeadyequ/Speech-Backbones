# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from einops import rearrange
import math
import torch
import torch.nn.functional as F
from GradTTS.model.base import BaseModule
from GradTTS.model.diffusion import *
from GradTTS.model.estimators import GradLogPEstimator2dCond

# diffuser
from src.diffusers import UNet2DConditionModel


class CondDiffusion(BaseModule):
    """
    Conditional Diffusion that denoising mel-spectrogram from latent normal distribution
    """

    def __init__(self,
                 n_feats,
                 dim,
                 n_spks=1,
                 spk_emb_dim=64,
                 emo_emb_dim=768,
                 beta_min=0.05,
                 beta_max=20,
                 pe_scale=1000,
                 data_type="melstyle",
                 att_type="linear"
                 ):
        """

        Args:
            n_feats:
            dim:   mel dim
            n_spks:
            spk_emb_dim:
            beta_min:
            beta_max:
            pe_scale:
            att_type:  "linear", "crossatt"
        """
        super(CondDiffusion, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.data_type = data_type
        self.att_type = att_type

        # choose unit model
        #if self.data_type == "melstyle" or self.data_type == "psd":
        if self.att_type == "linear":
            sample_channel_n = 3
        else:
            sample_channel_n = 1

        self.estimator = GradLogPEstimator2dCond(dim,
                                                 sample_channel_n=sample_channel_n,
                                                 n_spks=n_spks,
                                                 spk_emb_dim=spk_emb_dim,
                                                 pe_scale=pe_scale,
                                                 emo_emb_dim=emo_emb_dim,
                                                 att_type=att_type,
                                                 )

    def forward_diffusion(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0 * torch.exp(-0.5 * cum_noise) + mu * (1.0 - torch.exp(-0.5 * cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device,
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self,
                          z,
                          mask,
                          mu,
                          n_timesteps,
                          stoc=True,
                          spk=None,
                          psd=None,
                          melstyle=None,
                          emo_label=None,
                          align_len=None,
                          align_mtx=None
                          ):
        """
        Denoising process, given z, mu and conditioned on emolabel, psd, spk
        mel_len = 100 for length consistence
        Args:
            z:      (b, 80, mel_len)
            mask:   (b, 1, mel_len)
            mu:     (b, 80, mel_len)
            n_timesteps: int
            stoc:   bool, default = False
            spk:    (b, spk_emb) if exist else  NONE
            emo:    (b, emo_emb) if exist else  NONE
            psd:    (b, psd_dim, psd_len)   psd_len is not mel_len, psd_dim = 3 when eng/pitch/dur, psd=256 when wav2vector
            emo_label: (b, 1, emo_n)

        Returns:

        """
        h = 1.0 / n_timesteps
        xt = z * mask

        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype,
                                                   device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max,
                                cumulative=False)
            score_emo = self.estimator(
                x=xt,
                mask=mask,
                mu=mu,
                t=t,
                spk=spk,
                psd=psd,
                melstyle=melstyle,
                emo_label=emo_label,
                align_len=align_len,
                align_mtx=align_mtx
            )
            dxt_det = 0.5 * (mu - xt) - score_emo

            # adds stochastic term
            if stoc:
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - score_emo)
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def reverse_diffusion_interp_mod(
            self,
            z,
            mask,
            mu,
            n_timesteps,
            stoc=True,
            spk=None,
            melstyle1=None,
            emo_label1=None,
            melstyle2=None,
            emo_label2=None,
            align_len=None,
            align_mtx=None
    ):
        h = 1.0 / n_timesteps
        xt = z * mask

        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype,
                                                   device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max,
                                cumulative=False)

            # linear interpolate
            u = 0.5
            emo_label = emo_label1 * u + emo_label2 * (1 - u)

            if melstyle1.shape[1] > melstyle2.shape[1]:
                melstyle1 = melstyle1[:, :melstyle2.shape[1], :]
            else:
                melstyle2 = melstyle1[:, :melstyle1.shape[1], :]
            melstyle = melstyle1 * u + melstyle2 * (1 - u)

            score_emo = self.estimator(
                x=xt,
                mask=mask,
                mu=mu,
                t=t,
                spk=spk,
                melstyle=melstyle,
                emo_label=emo_label,
                align_len=align_len,
                align_mtx=align_mtx
            )
            dxt_det = 0.5 * (mu - xt) - score_emo

            # adds stochastic term
            if stoc:
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - score_emo)
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def reverse_diffusion_interp(self, z, mask, mu, n_timesteps,
                                 stoc=True,
                                 spk=None,
                                 emo1=None,
                                 emo2=None,
                                 psd1=None,
                                 psd2=None,
                                 emolabel1=None,
                                 emolabel2=None
                                 ):
        interpolate_rate = 0.5
        h = 1.0 / n_timesteps
        xt = z * mask

        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype,
                                                   device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max,
                                cumulative=False)

            if emo1 is not None:
                hidden_stats1 = emo1
            if psd1 is not None:
                hidden_stats1 = torch.concat(psd1, dim=1)

            if emo2 is not None:
                hidden_stats2 = emo1
            elif psd2 is not None:
                hidden_stats2 = torch.concat(psd1, dim=1)
            else:
                hidden_stats2 = None

            score_emo = self.estimator(xt,
                                       mask,
                                       mu,
                                       t,
                                       spk,
                                       hidden_stats1)
            # for interpolation
            if hidden_stats2 is not None:
                score_emo2 = self.estimator(xt, mask, mu, t, spk, hidden_stats2)
                score_emo = score_emo * interpolate_rate + score_emo2 * (1 - interpolate_rate)
            dxt_det = 0.5 * (mu - xt) - score_emo

            # adds stochastic term
            if stoc:
                ## add stochastics
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - score_emo)
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self,
                z,
                mask,
                mu,
                n_timesteps,
                stoc=False,
                spk=None,
                psd=None,
                melstyle=None,
                emo_label=None,
                align_len=None,
                align_mtx=None
                ):
        return self.reverse_diffusion(z=z,
                                      mask=mask,
                                      mu=mu,
                                      n_timesteps=n_timesteps,
                                      stoc=stoc,
                                      spk=spk,
                                      psd=psd,
                                      melstyle=melstyle,
                                      emo_label=emo_label,
                                      align_len=align_len,
                                      align_mtx=align_mtx
                                      )

    def loss_t(self,
               x0,
               mask,
               mu,
               t,
               spk=None,
               psd=None,
               melstyle=None,
               emo_label=None,
               align_len=None
               ):
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        noise_estimation = self.estimator(xt,
                                          mask,
                                          mu,
                                          t,
                                          spk=spk,
                                          psd=psd,
                                          melstyle=melstyle,
                                          emo_label=emo_label,
                                          align_len=align_len
                                          )
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z) ** 2) / (torch.sum(mask) * self.n_feats)
        return loss, xt

    def compute_loss(self,
                     x0,
                     mask,
                     mu,
                     offset=1e-5,
                     spk=None,
                     psd=None,
                     melstyle=None,
                     emo_label=None,
                     align_len=None,
                     ):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0,
                           mask,
                           mu,
                           t,
                           spk,
                           psd,
                           melstyle,
                           emo_label,
                           align_len
                           )