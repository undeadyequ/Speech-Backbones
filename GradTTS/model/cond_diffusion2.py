# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
import torch
import torchvision.transforms.functional as F
from GradTTS.model.base import BaseModule
from GradTTS.model.diffusion import *
from GradTTS.model.estimator import GradLogPEstimator2dCond, DiTMocha, STDit

class CondDiffusion(BaseModule):
    """
    Conditional Diffusion that denoising mel-spectrogram from latent normal distribution with Unet or Dit core
    """
    def __init__(self,
                 n_feats,
                 dim,
                 sample_channel_n=1,
                 n_spks=1,
                 spk_emb_dim=64,
                 emo_emb_dim=768,
                 beta_min=0.05,
                 beta_max=20,
                 pe_scale=1000,
                 data_type="melstyle",
                 att_type="linear",
                 att_dim=128,
                 heads=4,
                 p_uncond=0.2,
                 estimator_type="unet",  # dit or unet,
                 dit_mocha_config=None,
                 stdit_config=None,
                 stditMocha_config=None
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
        self.estimator_type = estimator_type

        # initiate estimates with input
        if self.estimator_type == "dit":
            kwargs = dit_mocha_config
            self.estimator = DiTMocha(**kwargs)
        elif self.estimator_type == "STDit":
            self.emo_mlp = torch.nn.Sequential(torch.nn.Linear(5, 128), Mish(),
                                               torch.nn.Linear(128, stdit_config["gin_channels"]))
            self.estimator = STDit(bAttn_type="base", **stdit_config)
        elif self.estimator_type == "STDitCross":
            self.emo_mlp = torch.nn.Sequential(torch.nn.Linear(5, 128), Mish(),
                                               torch.nn.Linear(128, stdit_config["gin_channels"]))
            self.estimator = STDit(bAttn_type="cross", **stditMocha_config)
        elif self.estimator_type == "STDitMocha":
            self.emo_mlp = torch.nn.Sequential(torch.nn.Linear(5, 128), Mish(),
                                               torch.nn.Linear(128, stdit_config["gin_channels"]))
            self.estimator = STDit(bAttn_type="mocha", **stditMocha_config)
        else:
            kwargs = None

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
                align_mtx=None,
                guidence_strength=3.0,
                attn_mask=None
                ):
        """
        Given z, mu and conditioning (emolabel, psd, spk), denoise melspectrogram by ODE or SDE solver (stoc)
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
            melstyle: (b, ?, ?)
            emo_label: (b, 1, emo_n)
        Returns:
        """
        h = 1.0 / n_timesteps
        xt = z * mask

        self_attns_list = []
        cross_attns_list = []
        attn_img_time_ind = (0, 10, 20, 30, 40, 49)

        if "STDit" in self.estimator_type:
            emo_label = torch.nn.functional.one_hot(emo_label, num_classes=5).to(torch.float)
            emo_label = self.emo_mlp(emo_label)
        for i in range(n_timesteps):
            #print("{}th time".format(i))
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, cumulative=False)

            if self.estimator_type == "dit":
                x = torch.stack([xt, mu], dim=1).transpose(2, 3)             # (b, 2, T, 80)
                #r = torch.concat([psd, melstyle], dim=1)     # (b, 80*2, T1)
                r = melstyle                                # (b, T1, 256)
                score_emo, attn_selfs, attn_crosses = self.estimator(
                    x=x,
                    y1=spk,
                    y2=emo_label,
                    r=r,
                    t=t)
                score_emo = score_emo[:, 0, :, :].transpose(1, 2)  ## ?? Tempt
            elif self.estimator_type == "STDit":
                # for stDit
                score_emo = self.estimator(
                    t=t, x=xt, mask=mask, mu=mu, c=emo_label)
                attn_selfs, attn_crosses = None, None
                # score_emo: (N, out_channels, H, W), attn_crosses: [t, block_n, b, HW?, T]
            # t: (b, ), xt: (b, mel_dim, cut_l), mask: (b, 1, cut_l), mu: (b, mel_dim, cut_l), emo_label:(b, emo_dim), melstyle:(b, mel_dim, cut_l)
            elif self.estimator_type == "STDitMocha" or self.estimator_type == "STDitCross":
                # for stDit
                score_emo, attn_crosses = self.estimator(
                    t=t, x=xt, mask=mask, mu=mu, c=emo_label, r=melstyle, attnCross=attn_mask)
                attn_selfs = None
            else:
                raise IOError("Estimator type is wrong")

            # Get dxt
            dxt_det = 0.5 * (mu - xt) - score_emo
            if self.estimator_type == "STDitMocha" or self.estimator_type == "STDitCross":
                if i in attn_img_time_ind:
                    self_attns_list.append(attn_selfs)
                    cross_attns_list.append(attn_crosses)

            ## adds stochastic term
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

        #self_attns = torch.stack(self_attns_list, dim=0)
        cross_attns = torch.stack(cross_attns_list, dim=0)
        return xt, self_attns_list, cross_attns

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
                     attnCross=None
                     ):
        """

        Args:
            x0,       # (b, mel_dim, cut_l)
            mask,  # (b, )
            mu,    # (b, mel_dim, cut_l)
            offset,
            spk: # (b, )
            psd: # (b, )
            melstyle:  # (b, mel_dim, cut_l)
            emo_label:      # (b, )
        Returns:
        """
        t = torch.rand(x0.shape[0],
                       dtype=x0.dtype,
                       device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        xt, z = self.forward_diffusion(x0, mask, mu, t)  #
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)

        # Concatenate condition and input

        # Estimate noise
        if self.estimator_type == "dit":
            # *TEST1*
            x = torch.stack([xt, mu], dim=1).transpose(2, 3)  # (b, 2, T, 80)
            noise_estimation, attn_selfs, attn_crosses = self.estimator(
                x=x,
                y1=spk,
                y2=emo_label,
                r=melstyle,
                t=t)  # (b, T, 80)
            noise_estimation = noise_estimation[:, 0, :, :]  ## ?? Tempt
            noise_estimation = noise_estimation.transpose(1, 2)
            noise_estimation = noise_estimation * mask
            attn_selfs, attn_crosses = None, None
        elif self.estimator_type == "STDit":
            emo_label = torch.nn.functional.one_hot(emo_label, num_classes=5).to(torch.float)
            emo_label = self.emo_mlp(emo_label)
            # t: (b, ), xt: (b, mel_dim, cut_l), mask: (b, 1, cut_l), mu: (b, mel_dim, cut_l), emo_label:(b, emo_dim), melstyle:(b, mel_dim, cut_l)
            noise_estimation = self.estimator(t=t, x=xt, mask=mask, mu=mu, c=emo_label, r=melstyle)
            attn_selfs, attn_crosses= None, None
        elif self.estimator_type == "STDitMocha":
            emo_label = torch.nn.functional.one_hot(emo_label, num_classes=5).to(torch.float)
            emo_label = self.emo_mlp(emo_label)
            # t: (b, ), xt: (b, mel_dim, cut_l), mask: (b, 1, cut_l), mu: (b, mel_dim, cut_l), emo_label:(b, emo_dim), melstyle:(b, mel_dim, cut_l)
            noise_estimation, attn_crosses = self.estimator(t=t, x=xt, mask=mask, mu=mu, c=emo_label, r=melstyle,
                                                            attnCross=attnCross)
            attn_selfs = None
        else:
            print("Wrong estimation")
            noise_estimation, attn_selfs, attn_crosses = None, None, None

        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z) ** 2) / (torch.sum(mask) * self.n_feats)
        #loss = mean_flat((noise_estimation - z) ** 2).mean()      ######## TEST for pure normalize loss ########
        return loss, xt, attn_selfs, attn_crosses

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))