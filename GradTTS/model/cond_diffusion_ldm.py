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
from GradTTS.model.base import BaseModule
from GradTTS.model.diffusion import *
from GradTTS.model.estimators import GradLogPEstimator2dCond

from GradTTS.model.text_encoder import TextEncoder

# diffuser
from src.diffusers import UNet2DConditionModel
from GradTTS.text.symbols import symbols
from typing import Any, Dict, List, Optional, Tuple, Union
from GradTTS.model.utils import (sequence_mask, generate_path, duration_loss,
                                 fix_len_compatibility, align_a2b)

add_blank = True
nsymbols = len(symbols) + 1 if add_blank else len(symbols)


class UNet2DConditionModelEMO(UNet2DConditionModel):
    """
    A extending unet with more conditioning value including spk and psd
    """
    def __init__(self,
                 n_spks,
                 spk_emb_dim,
                 psd_n,
                 n_feats
                 ):
        super(UNet2DConditionModel, self).__init__()
        self.unet_model = UNet2DConditionModel(
            sample_size=(80, 100),
            in_channels=1,   # Number of channels in the input sample
            out_channels=1,  #
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            mid_block_type="UNetMidBlock2DCrossAttn",
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            encoder_hid_dim=165,  # ?
            encoder_hid_dim_type="text_proj",
            #cross_attention_dim=256,
            cross_attention_dim=32,
            block_out_channels=(32, 64),
            norm_num_groups=4,
            layers_per_block=2
            #class_embed_type="simple_projection",
            #addition_embed_type="image",
            #projection_class_embeddings_input_dim=5,
            #class_embeddings_concat=True
        )

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        sampleadd: str = "concate",
        guidance_scale: float = 3.0

    ):
        """

        Args:
            sample:
            timestep:
            encoder_hidden_states:
            class_labels:
            timestep_cond:
            attention_mask:
            cross_attention_kwargs:
            added_cond_kwargs:
            down_block_additional_residuals:
            mid_block_additional_residual:
            encoder_attention_mask:
            return_dict:
            spk:
            psd:
            sampleadd:
            guidance_scale:
            --------------Conditional--------------------
            enc_hids: # (b, pmel_len, mel_dim)
            spk:      (b, spk_emb) if exist else NONE
            psd:      (b, pmel_len, psd_dim)    # psd has been aligned to pmel_len
            emo_label: (b, 1, emo_n)
            ---------------------------------------------


        Returns:

        """
        sample = sample.unsqueeze(1)

        # Get conditional embeddings (emo_label, psd, melstyle, text)
        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            encoder_hidden_uncond = torch.zeros_like(encoder_hidden_states, device=encoder_hidden_states.device)
            # combine uncond and cond enc_hid
            encoder_hidden_states = torch.concat([encoder_hidden_states, encoder_hidden_uncond])

            sample = torch.cat([sample] * 2) if do_classifier_free_guidance else sample
            timestep = torch.cat([timestep] * 2) if do_classifier_free_guidance else timestep
            encoder_attention_mask = torch.cat([encoder_attention_mask] * 2)

        # get noise residual
        noise_pred = self.unet_model(
            sample=sample,  # (batch, height, width)
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,  # (batch, sequence_length, feature_dim)
            encoder_attention_mask=encoder_attention_mask,
        )
        noise_pred = noise_pred["sample"].squeeze(1)

        if do_classifier_free_guidance:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        return noise_pred


class CondDiffusionLDM(BaseModule):
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
        super(CondDiffusionLDM, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.att_type = att_type

        self.estimator = UNet2DConditionModelEMO(
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
            psd_n=3,
            n_feats=n_feats
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
                          enc_hids=None,
                          enc_hids_mask=None
                          ):
        """
        Denoising mel from z and mu, conditioned on
        class labels: emolabel, spk, psd, encoder_hidden_states
        (where sequential psd and encoder_hidden_states have been aligned)

        mel_len = 100 for length consistence
        Args:
            z:        (b, pmel_len, 80)
            mask:     (b, )
            mu:       0
            n_timesteps: int
            stoc:   bool, default = False
            --------------Conditional--------------------
            enc_hids: # (b, pmel_len, mel_dim)
            spk:      (b, pmel_len, spk_emb) if exist else NONE
            psd:      (b, pmel_len, psd_dim)    # psd has been aligned to pmel_len
            emo_label: (b, 1, emo_n)
            ---------------------------------------------
            enc_hids_mask: # (b,)
            emo:      Not used
        Returns:
            xt:     (b, mel_len, 80)
        """
        h = 1.0 / n_timesteps
        xt = z * mask

        for i in range(n_timesteps):
            # com
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype,
                                                   device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max,
                                cumulative=False)
            score_emo = self.estimator(
                sample=xt,      # (batch, channel, height, width)
                timestep=t,
                encoder_hidden_states=enc_hids,  # (batch, sequence_length, feature_dim)
                encoder_attention_mask=torch.squeeze(enc_hids_mask, 2),
            )
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
            xt = (xt - dxt) * mask.squeeze(1)
        return xt

    @torch.no_grad()
    def reverse_diffusion_interp(self,
                                 z,
                                 mask,
                                 mu,
                                 n_timesteps,
                                 stoc=True,
                                 spk=None,
                                 emo1=None,
                                 emo2=None,
                                 psd1=None,
                                 psd2=None,
                                 emolabel1=None,
                                 emolabel2=None
                                 ):
        """
        Interpolate two emolabels, z, or psd.
        Args:
            z ():
            mask ():
            mu ():
            n_timesteps ():
            stoc ():
            spk ():
            emo1 ():
            emo2 ():
            psd1 ():
            psd2 ():
            emolabel1 ():
            emolabel2 ():

        Returns:

        """
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

            # Combine sample and spk
            spk = self.spk_mlp(self.spk_emb(spk))
            spk = spk.unsqueeze(1).repeat(1, xt.shape[1], 1)
            xt = torch.stack([xt, spk], 1)

            # Interpolate between score
            score_emo1 = self.estimator(
                sample=xt,
                timestep=t,
                encoder_hidden_states=hidden_stats1,
                class_labels=emolabel1,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                encoder_attention_mask=text_mask
            )
            score_emo2 = self.estimator(
                sample=xt,
                timestep=t,
                encoder_hidden_states=hidden_stats1,
                class_labels=emolabel1,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                encoder_attention_mask=text_mask
            )

            score_emo = score_emo1 * interpolate_rate + score_emo2 * (1 - interpolate_rate)
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
                enc_hids=None,
                enc_hids_mask=None
                ):
        return self.reverse_diffusion(z=z,
                                      mask=mask,
                                      mu=mu,
                                      n_timesteps=n_timesteps,
                                      stoc=stoc,
                                      enc_hids=enc_hids,
                                      enc_hids_mask=enc_hids_mask
                                      )

    def loss_t(self,
               x0,
               mask,
               mu,
               t,
               enc_hids=None,
               enc_hids_mask=None
               ):
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)

        noise_estimation = self.estimator(
            sample=xt,  # (batch, channel, height, width)
            timestep=t,
            encoder_hidden_states=enc_hids,  # (batch, sequence_length, feature_dim)
            encoder_attention_mask=torch.squeeze(enc_hids_mask, 2),
        )

        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z) ** 2) / (torch.sum(mask) * self.n_feats)
        return loss, xt

    def compute_loss(self,
                     x0,
                     mask,
                     mu,
                     offset=1e-5,
                     enc_hids=None,
                     enc_hids_mask=None
                     ):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(
            x0=x0,
            mask=mask,
            mu=mu,
            t=t,
            enc_hids=enc_hids,
            enc_hids_mask=enc_hids_mask
        )