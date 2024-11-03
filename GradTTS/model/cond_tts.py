# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random

import torch
from GradTTS.model import monotonic_align
from GradTTS.model.base import BaseModule
from GradTTS.model.text_encoder import TextEncoder
from GradTTS.model.diffusion import Diffusion, Mish
from GradTTS.model.cond_diffusion import CondDiffusion
from GradTTS.model.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility
from GradTTS.model.utils import (sequence_mask, generate_path, duration_loss,
                                 fix_len_compatibility, align_a2b, align, align_a2b_padcut, cut_pad_start_end)
from GradTTS.model.maskCreation import create_p2p_mask, DiffAttnMask

ADD_COND_TO_ENC = True
USE_MUY = True


class CondGradTTS(BaseModule):
    def __init__(self, n_vocab, n_spks, spk_emb_dim, emo_emb_dim,
                 n_enc_channels, filter_channels, filter_channels_dp,
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                 n_feats, dec_dim, sample_channel_n, beta_min, beta_max, pe_scale, unet_type,
                 att_type, att_dim, heads, p_uncond,
                 psd_n=3,
                 melstyle_n=768):
        super(CondGradTTS, self).__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.unet_type = unet_type
        self.att_type = att_type

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        #self.emo_emb = torch.nn.Embedding(5, emo_emb_dim)
        self.spk_mlp = torch.nn.Sequential(torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4), Mish(),
                                           torch.nn.Linear(spk_emb_dim * 4, n_feats))
        self.emo_mlp = torch.nn.Sequential(torch.nn.Linear(emo_emb_dim, emo_emb_dim), Mish(),
                                           torch.nn.Linear(emo_emb_dim, n_feats))
        self.psd_mlp = torch.nn.Sequential(torch.nn.Linear(psd_n, psd_n * 4), Mish(),
                                           torch.nn.Linear(psd_n * 4, n_feats))
        self.encoder = TextEncoder(n_vocab, n_feats, n_enc_channels, 
                                   filter_channels, filter_channels_dp, n_heads, 
                                   n_enc_layers, enc_kernel, enc_dropout, window_size)
        self.decoder = CondDiffusion(n_feats, dec_dim,
                                     sample_channel_n,
                                     n_spks,
                                     spk_emb_dim, emo_emb_dim,
                                     beta_min, beta_max, pe_scale, unet_type,
                                     att_type,
                                     att_dim, heads, p_uncond)
        self.melstyle_mlp = torch.nn.Sequential(torch.nn.Linear(melstyle_n, melstyle_n), Mish(),
                                           torch.nn.Linear(melstyle_n, n_feats))

    def predict_prior(self,
                      x,
                      x_lengths,
                      spk,
                      length_scale
                      ):
        """
        Predict prior (mu_y) from x by encoder which encoder x to mu_x while predict logw.
        """
        # Get mu_x and logw (log-scaled token durations)
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale

        # Get y_lengths (predicted mel len) from logw
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # construct attn (alignment map ) by y_mask (predicted mel len) and x_mask
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Get mu_y by attnb and mu_x
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        return mu_y, y_mask, y_max_length, attn, attn_mask

    @torch.no_grad()
    def forward(self,
                x,
                x_lengths,
                n_timesteps,
                temperature=1.0,
                stoc=False,
                spk=None,
                length_scale=1.0,
                psd=None,
                emo_label=None,
                melstyle=None,
                guidence_strength=3.0,
                attn_hardMask=None
                ):
        """
        Generates mel-spectrogram by encoder, decoder, from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        Encoder: Given text, encoder text (mu_y) with duration prediction
        Decoder: Given mu_y, spk, emo (emo style embedding), emo_label, denoising mel_spectrogram with time step loop

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
            attn_hardMask: [b,1], [c,1], l_q, [l_k, 1]
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        # Embed condition
        if spk is not None:
            spk = self.spk_mlp(self.spk_emb(spk))
        if emo_label is not None:
            emo_label = emo_label.to(torch.float)
            emo_label = self.emo_mlp(emo_label)   # (b, 1, 80)
        if melstyle is not None:
            melstyle = self.melstyle_mlp(melstyle.transpose(1, 2))

        # get prior
        mu_y, y_mask, y_max_length, attn, attn_mask = self.predict_prior(x, x_lengths, spk, length_scale)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Get sample latent representation from terminal distribution N(mu_y, I)
        variance = torch.randn_like(mu_y, device=mu_y.device) / temperature
        z = mu_y + variance

        if attn_hardMask is not None:
            #attn_hardMask_max = torch.max(attn_hardMask)
            attn_hardMask = torch.ones_like(attn_hardMask) - attn_hardMask
            attn_hardMask = torch.matmul(attn.transpose(2, 3), attn_hardMask)  # (1,1, mu_y/l_q, l_k)
            attn_hardMask = attn_hardMask.unsqueeze(3).repeat(1, 1, 1, 80, 1)  # (1,1, l_q, 80, l_k)
            attn_hardMask = attn_hardMask.view(1, 1, -1, attn_hardMask.shape[-1]) # (1,1, l_q * 80, l_k)
            #attn_hardMask_max = torch.max(attn_hardMask)
            attn_hardMask = attn_hardMask>0
            attn_hardMask = ~attn_hardMask

        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z,
                                       y_mask,
                                       mu_y,
                                       n_timesteps,
                                       stoc=stoc,
                                       spk=spk,
                                       psd=psd,
                                       melstyle=melstyle,
                                       emo_label=emo_label,
                                       align_len=mu_y.shape[-1],
                                       align_mtx=attn,
                                       guidence_strength=guidence_strength,
                                       attn_mask=attn_hardMask
                                       )
        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    @torch.no_grad()
    def reverse_diffusion_mix(
            self,
            x,
            x_lengths,
            n_timesteps,
            temperature=1.0,
            stoc=False,
            length_scale=1.0,
            mix_mode="noise",

            spk1=None,
            emo_label1=None,
            melstyle1=None,
            attn_hardMask1=None,

            spk2=None,
            emo_label2=None,
            melstyle2=None,
            attn_hardMask2=None,

            guidence_strength=3.0,
            ref2_start_p=None,
            ref2_end_p=None,
            tgt_start_p=None,
            tgt_end_p=None,
    ):
        """
        Reverse diffusion process with phoneme2phoneme style transfer given 2 mode (mix_mode):
        1. noise mode (ref2_y is not null)
            mu_y = f(x)[,,, tgt_start:tgt_end] <- diffuse(ref2_y)[..., ref_start:ref_end]
            z = mu_y + norm
            mu_y = f(x)[,,, tgt_start:tgt_end]
            z = mu_y + norm <- mu_y + norm[..., !tgt_start:tgt_end] + diffuse(ref2_y)[..., ref_start:ref_end]

        2. crossAttn mode (attn_hardMask is not null)
            mu_y = f(x)
            z = mu_y * norm
            attn_map = attn_map * attn_hardMask

        reversion diffusion
            dXt = (mu_y - xt - s(xt, u, t))bt * dt + bt^2 * dwt
            x(T-1) = mu_y + noise(xT)
            X(T-2) = x(T-1) + noise(x(T-1))
        Args:
        Returns:
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        # Embed condition
        spk1 = self.spk_mlp(self.spk_emb(spk1))
        emo_label1 = self.emo_mlp(emo_label1.to(torch.float))   # (b, 1, 80)
        melstyle1 = self.melstyle_mlp(melstyle1.transpose(1, 2))

        spk2 = self.spk_mlp(self.spk_emb(spk2))
        emo_label2 = self.emo_mlp(emo_label2.to(torch.float))  # (b, 1, 80)
        melstyle2 = self.melstyle_mlp(melstyle2.transpose(1, 2))

        # get prior (mu_y1) from x
        mu_y1, y1_mask, y1_max_length, attn1, attn1_mask = self.predict_prior(x, x_lengths, spk1, length_scale)  # ?? Need spk1 ??
        encoder_outputs = mu_y1[:, :, :y1_max_length]  # ?? Needed ??

        ## 1. Reshape size of attn_hardMask (mu_x * ref_pFrame_len) to (80 * mu_y, ref_pFrame_len) by attention map
        ## 2. Change 1 to Boolen
        attn_hardMask = torch.ones_like(attn_hardMask1) - attn_hardMask1
        attn_hardMask = torch.matmul(attn1.transpose(2, 3), attn_hardMask)  # (1,1, mu_y/l_q, l_k)
        attn_hardMask = attn_hardMask.unsqueeze(3).repeat(1, 1, 1, 80, 1)  # (1,1, l_q, 80, l_k)
        attn_hardMask = attn_hardMask.view(1, 1, -1, attn_hardMask.shape[-1])  # (1,1, l_q * 80, l_k)
        # attn_hardMask_max = torch.max(attn_hardMask)
        attn_hardMask = attn_hardMask > 0

        # tempt check
        #attn_hardMask = attn_hardMask > -100 # all true

        attn_hardMask1 = attn_hardMask   # transfer phone = True
        attn_hardMask2 = ~attn_hardMask  # transfer phone = False


        torch.set_printoptions(threshold=100000)
        #print("if there is true")
        #print((attn_hardMask1[0, 0, :, :]==True).nonzero(as_tuple=True))
        #print(attn_hardMask1[0, 0, :, :])

        # p2p_mode == noise
        if mix_mode == "noise":
            # diffusing ref2_y
            offset = 1e-2
            t_T = torch.ones(
                mu_y1.shape[0],
                dtype=mu_y1.dtype,
                device=mu_y1.device)
            t_T = torch.clamp(t_T, offset, 1.0 - offset)
            ref2_y = align_a2b_padcut(melstyle2, mu_y1.shape[2])
            diffused_ref2 = self.decoder.diffuse_x0(ref2_y, t_T)

            ## Align frame-level ref2_start/end_p to mu_y1 (as ref2 should same with mu_y1 for diffusing)
            ref2_start_p_mod, ref2_end_p_mod = cut_pad_start_end(
                ref2_start_p, ref2_end_p,
                sr_seq_len=ref2_y.shape[2]
                if len(ref2_y.shape) == 3
                else ref2_y.shape[1], tr_seq_len=mu_y1.shape[2]
            )
            # Align phoneme-level tgt_start/end_p to mu_y1 (as mu_x is converted to mu_y)  ?? How to ??
            tgt_start_p_mod, tgt_end_p_mod = get_first_last_nonZero(attn1, tgt_start_p, tgt_end_p)

            # Guarantee tgt range to same as ref range
            tgt_p_mod_range = tgt_end_p_mod - tgt_start_p_mod
            ref2_p_mod_range = ref2_end_p_mod - ref2_start_p_mod
            if tgt_p_mod_range != ref2_p_mod_range:
                tgt_end_p_mod = tgt_start_p_mod + ref2_p_mod_range

            # (Check) -> should be similar to norm distr
            #dr_min, dr_max = torch.min(diffused_ref2), torch.max(diffused_ref2)

            # guarantee ref2 phoneme is not cut
            if ref2_end_p > mu_y1.shape[2]:
                raise IOError("The ref2_start {} and ref2_end {} should not be greater than given mu_y {}".format(
                    ref2_start_p_mod, ref2_end_p_mod, mu_y1.shape[2]))
            #mu_y1[:, :, ref2_start_p:ref2_end_p] = diffused_y2[:, :, ref2_start_p:ref2_end_p]  # (b, 80, L)

            # initiate normal noise N(mu_y, I)
            variance = torch.randn_like(mu_y1, device=mu_y1.device) / temperature
            # insert p2p noise
            variance[:, :, tgt_start_p_mod:tgt_end_p_mod] = diffused_ref2[:, :, ref2_start_p_mod:ref2_end_p_mod]  # (b, 80, L)
            #variance[:, :, tgt_start_p_mod:tgt_end_p_mod] = torch.tensor(0.1, device=variance.device)

            # variance[:, :, ref2_start_p:ref2_end_p] = diffused_ref2[:, :, ref2_start_p:ref2_end_p]  # (b, 80, L)
        else:  # p2p_mode is crossAttn
            variance = torch.randn_like(mu_y1, device=mu_y1.device) / temperature

        z = mu_y1 + variance
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder.reverse_diffusion_mix(
            z,
            y1_mask,
            mu_y1,
            n_timesteps,
            stoc=stoc,
            spk1=spk1,
            emo_label1=emo_label1,
            melstyle1=melstyle1,
            attn_hardMask1=attn_hardMask1,
            spk2=spk2,
            emo_label2=emo_label2,
            melstyle2=melstyle2,
            attn_hardMask2=attn_hardMask2,
            align_len=mu_y1.shape[-1],
            align_mtx=attn1,
            guidence_strength=guidence_strength)

        decoder_outputs = decoder_outputs[:, :, :y1_max_length]
        return encoder_outputs, decoder_outputs, attn1[:, :, :y1_max_length], attn_hardMask

    #@torch.no_grad()
    def reverse_diffusion_interp(self,
                                 x,
                                 x_lengths,
                                 n_timesteps,
                                 temperature=1.0,
                                 stoc=False,
                                 spk=None,
                                 length_scale=1.0,
                                 emo_label1=None,
                                 melstyle1=None,
                                 pitch1=None,
                                 emo_label2=None,
                                 melstyle2=None,
                                 pitch2=None,
                                 interp_type="simple",
                                 mask_time_step=None,
                                 mask_all_layer=True,
                                 temp_mask_value=0,
                                 guidence_strength=3.0
                                 ):
        x, x_lengths = self.relocate_input([x, x_lengths])

        # Embed condition
        if spk is not None:
            spk = self.spk_mlp(self.spk_emb(spk))
        emo_label1 = self.emo_mlp(emo_label1.to(torch.float)) # (b, 1, 80)
        emo_label2 = self.emo_mlp(emo_label2.to(torch.float))  # (b, 1, 80)
        melstyle1 = self.melstyle_mlp(melstyle1.transpose(1, 2))
        melstyle2 = self.melstyle_mlp(melstyle2.transpose(1, 2))

        # Get `mu_x` and `logw` (log-scaled token durations)
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale

        # Get y_lengths (predicted mel len) from logw
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # construct `attn` (alignment map ) by y_mask (predicted mel len) and x_mask
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Get mu_y by attnb and mu_x
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Get sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics

        decoder_outputs = self.decoder.reverse_diffusion_interp_mod(
            z,
            y_mask,
            mu_y,
            n_timesteps,
            stoc=stoc,
            spk=spk,
            melstyle1=melstyle1,
            emo_label1=emo_label1,
            pitch1=pitch1,
            melstyle2=melstyle2,
            emo_label2=emo_label2,
            pitch2=pitch2,
            align_len=melstyle1.shape[1],  # ?? mu_y.shape[-1] ??
            align_mtx=attn,
            interp_type=interp_type,
            mask_time_step=mask_time_step,
            mask_all_layer=mask_all_layer,
            temp_mask_value=temp_mask_value,
            guidence_strength=guidence_strength
        )
        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def compute_loss(self,
                     x,
                     x_lengths,
                     y,
                     y_lengths,
                     spk=None,
                     out_size=None,
                     emo=None,
                     psd=None,
                     melstyle=None,
                     emo_label=None
                     ):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        # spk, emo, mel_style embed condition
        if spk is not None:
            spk = self.spk_mlp(self.spk_emb(spk))
        if emo_label is not None:
            emo_label = emo_label.to(torch.float)
            emo_label = self.emo_mlp(emo_label)   # (b, 1, 80)
        if melstyle is not None:
            melstyle = self.melstyle_mlp(melstyle.transpose(1, 2))

        # Get mu_x (encoded x) and logw (log-scaled token durations)
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use gd logw_ by attn which predicted by MAS (find most likely alignment `attn` between text and mel-spectrogram)
        with torch.no_grad(): 
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const
            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        # Compute duration loss between logw and logw_
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # 4. Align sequential and fixed length hidden states (to the length of the predicted phoneme-level)
        """
        
        align_target_len = mu_x.shape[-1] if ADD_COND_TO_ENC else log_prior.shape[-1]
        align_target_mask = x_mask if ADD_COND_TO_ENC else y_mask
        enc_hid_cond, enc_hids_mask = self.align_combine_cond(
            psd, melstyle, spk, emo_label,
            align_target_len, align_target_mask, attn
        )
        """
        # Align melstyle to mu_y length
        melstyle_align = align_a2b_padcut(melstyle, y.shape[-1], attn, condtype="seq")
        #melstyle_mask = y_mask

        # Cut a small segment of mel-spectrogram (to increase batch size)
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            ## randomly select start and end point
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)
            
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
            #enc_hid_cond_cut = torch.zeros(enc_hid_cond.shape[0], enc_hid_cond.shape[1], out_size,
            #                               dtype=enc_hid_cond.dtype, device=enc_hid_cond.device)
            melstyle_cond_cut = torch.zeros(melstyle_align.shape[0], melstyle_align.shape[1], out_size,
                                           dtype=melstyle_align.dtype, device=melstyle_align.device)

            y_cut_lengths = []
            melstyle_lengths = []
            #max_x_cut_length = 1
            for i, (y_, out_offset_, enc_hid_cond_) in enumerate(zip(y, out_offset, melstyle_align)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]

                # cut enc_hids
                """
                cut_lower_x, cut_upper_x = get_cut_range_x(attn[i], cut_lower, cut_upper)
                x_cut_length = cut_upper_x - cut_lower_x
                if x_cut_length > max_x_cut_length:
                    max_x_cut_length = x_cut_length
                """
                melstyle_lengths.append(y_cut_length)
                melstyle_cond_cut[i, :, :y_cut_length] = enc_hid_cond_[:, cut_lower:cut_upper]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths, max_length=out_size).unsqueeze(1).to(y_mask)   # Set length to out_size to enable fix length learning
            #melstyle_lengths = torch.LongTensor(melstyle_lengths)
            #melstyle_mask = sequence_mask(melstyle_lengths, max_length=max_x_cut_length).unsqueeze(1).to(x_mask)
            #melstyle_mask = sequence_mask(melstyle_lengths, max_length=out_size).unsqueeze(1).to(x_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask
            melstyle_cond = melstyle_cond_cut

        # Get mu_y by attn
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y,
                                                  y_mask,
                                                  mu_y,
                                                  offset=1e-5,
                                                  spk=spk,
                                                  psd=psd,
                                                  melstyle=melstyle_cond,
                                                  emo_label=emo_label,
                                                  align_len=mu_y.shape[-1],
                                                  #align_mtx=attn
                                                  #enc_hids=enc_hid_cond.permute(0, 2, 1),
                                                  #enc_hids_mask=enc_hids_mask.permute(0, 2, 1)
                                                  )
        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        
        return dur_loss, prior_loss, diff_loss

    """
    def align_combine_cond(
            self,
            psd,
            melstyle,
            spk,
            emo_label,
            align_target_len,
            align_target_mask,
            attn
    ):
        psd_aligned = None
        if psd[0] is not None:
            psd = torch.stack(psd, 1).permute(0, 2, 1)  # (b, psd_dim, mel_len)
            psd = self.psd_mlp(psd)
            psd_aligned = align_a2b(psd.transpose(1, 2),
                                    align_target_len,
                                    attn.squeeze(1).transpose(1, 2))
        if melstyle is not None:
            melstyle = self.melstyle_mlp(melstyle.transpose(1, 2))
            psd_aligned = align_a2b(melstyle.transpose(1, 2),
                                    align_target_len,
                                    attn.squeeze(1).transpose(1, 2))  # (b, mel_dim, mel_len)
        spk_align = spk.unsqueeze(2).repeat(1, 1, align_target_len)  # (b, mel_dim, pmel_len)

        if emo_label is not None:
            emo_label = emo_label.to(torch.float)
            emo_label_align = emo_label.unsqueeze(2).repeat(1, 1, align_target_len)
            enc_hid_cond = torch.concat([psd_aligned, spk_align, emo_label_align], dim=1)  # # (b, all_dim, phnm_len)
        else:
            enc_hid_cond = torch.concat([psd_aligned, spk_align], dim=1)  # # (b, all_dim, phnm_len)
        enc_hids_mask = align_target_mask  # (b, )
        return enc_hid_cond, enc_hids_mask
    """


def get_cut_range_x(attn, cut_lower, cut_upper):
    cut_lower_x = (attn[:, cut_lower] == 1).nonzero(as_tuple=False)
    cut_upper_x = (attn[:, cut_upper-1] == 1).nonzero(as_tuple=False)
    if cut_upper_x.size()[0] == 0:
        torch.set_printoptions(threshold=10_000)
        print(attn.size(), cut_lower, cut_upper)
        print(attn[:, cut_upper])
        print(attn)
    cut_lower_x = cut_lower_x[0][0]
    cut_upper_x = cut_upper_x[0][0]
    #print("cut_lower_x, cut_upper_x:{}, {}".format(cut_lower_x, cut_upper_x))
    return cut_lower_x, cut_upper_x


def get_first_last_nonZero(x2y_attn, x_index_start, x_index_end):
    """
    get y_index at x2y_attn[x_index, :] where is the first/last nonzero.
    Args:
        x2y_attn (1, 1, x_len, y_len):
        x_index_start:
        x_index_end:
    Returns:

    """
    if len(x2y_attn.shape) == 4 and x2y_attn.shape[0] == 1 and x2y_attn.shape[1] == 1:
        y_nonzero_index_start = (x2y_attn[0, 0, x_index_start, :] == 1).nonzero(as_tuple=True)
        y_nonzero_index_start = y_nonzero_index_start[0][0]  # 0: 3rd dim, 0: first nonzero
        y_nonzero_index_end = (x2y_attn[0, 0, x_index_end, :] == 1).nonzero(as_tuple=True)
        y_nonzero_index_end = y_nonzero_index_end[0][-1] # 0: 3rd dim, 0: last nonzero
    else:
        raise IOError("x2y_attn should have 4 dims and 1 length for first two dims!")
    return y_nonzero_index_start, y_nonzero_index_end