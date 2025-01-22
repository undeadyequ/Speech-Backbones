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
from GradTTS.model.encoder.text_encoder import TextEncoder
from GradTTS.model.cond_diffusion_ldm import CondDiffusionLDM
from GradTTS.model.diffusion import Mish

from GradTTS.model.utils import (sequence_mask, generate_path, duration_loss,
                                 fix_len_compatibility, align_a2b)

ADD_COND_TO_ENC = True
USE_MUY = True


class CondGradTTSLDM(BaseModule):
    """
    Conditionable LDM gradTTS
    """
    def __init__(self,
                 n_vocab,
                 n_spks,
                 spk_emb_dim,
                 emo_emb_dim,
                 n_enc_channels,
                 filter_channels,
                 filter_channels_dp,
                 n_heads,
                 n_enc_layers,
                 enc_kernel,
                 enc_dropout,
                 window_size,
                 n_feats,
                 dec_dim,
                 beta_min,
                 beta_max,
                 pe_scale,
                 att_type,
                 interp_type="salient",
                 psd_n=3,
                 melstyle_n=768
                 ):
        """

        Args:
            n_vocab ():
            n_spks ():
            spk_emb_dim ():
            emo_emb_dim ():
            n_enc_channels ():
            filter_channels ():
            filter_channels_dp ():
            n_heads ():
            n_enc_layers ():
            enc_kernel ():
            enc_dropout ():
            window_size ():
            n_feats ():
            dec_dim ():
            beta_min ():
            beta_max ():
            pe_scale ():
            unet_type ():
            att_type ():
            interp_type ():
            if salient:
                Merge salient part of each mel
            if all:
                all part interpolation
        """
        super(CondGradTTSLDM, self).__init__()
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
        self.interp_type = interp_type
        self.melstyle_n = melstyle_n

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.spk_mlp = torch.nn.Sequential(torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4), Mish(),
                                           torch.nn.Linear(spk_emb_dim * 4, n_feats))
        self.psd_mlp = torch.nn.Sequential(torch.nn.Linear(psd_n, psd_n * 4), Mish(),
                                           torch.nn.Linear(psd_n * 4, n_feats))
        self.melstyle_mlp = torch.nn.Sequential(torch.nn.Linear(melstyle_n, melstyle_n), Mish(),
                                           torch.nn.Linear(melstyle_n, n_feats))
        # self.emo_emb = torch.nn.Embedding(5, emo_emb_dim) <- include in unet_model

        self.encoder = TextEncoder(n_vocab, n_feats, n_enc_channels,
                                   filter_channels, filter_channels_dp, n_heads,
                                   n_enc_layers, enc_kernel, enc_dropout, window_size)
        self.decoder = CondDiffusionLDM(n_feats, dec_dim, n_spks,
                                     spk_emb_dim, emo_emb_dim,
                                     beta_min, beta_max, pe_scale,
                                     att_type)
        #self.saDetect = SalientAreaDetector()
    @torch.no_grad()
    def forward(self,
                x,
                x_lengths,
                n_timesteps,
                temperature=1.0,
                stoc=False,
                spk=None,
                length_scale=1.0,
                emo=None,
                psd=None,
                emo_label=None,
                ref_speech=None,
                melstyle=None,
                ):
        """
        Generates mel-spectrogram from text, conditioning on below(No interpolations)
            1. emo_emb or emo_label
            2. ref_speech
            3. spk
        by following process:
        1. Get z with predicted mel length from x
        2. Sample z from normal
        3. Embed fixed length embedding
        4. align sequential hidden states
        5. Encode encoder hidden states
        Returns:
            1. encoder outputs
            2. decoder outputs (b, mel_dim, mel_len)
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
            spk:    (b, spk_emb) if exist else  NONE
            emo:    (b, emo_emb) if exist else  NONE
            psd:    tuple((b, psd_len' ,...,...)   psd_len is not mel_len, psd_dim = 3 when eng/pitch/dur, psd=256 when wav2vector
            emo_label: (b, 1, emo_n)
            melstyle: (b, mel_len, mel_dim)

        """
        x, x_lengths = self.relocate_input([x, x_lengths])
        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)
        # 1. Get predicted length of Z (latent representation) from given x by an encoder (include duration prediction)
        ## Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)  # mu_x: (b, mel_dim, phnm_len), logw: (b, 1, phnm_len)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        ## Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)  # (b, 1, pmel_len)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)                      # (b, 1, phnm_len, pmel_len)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)  # (b, 1, phnm_len, pmel_len)

        ## Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))  #
        mu_y = mu_y.transpose(1, 2)                                      # (b, mel_dim, pmel_len)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # 2. Sample Z from terminal distribution N(mu_y, I) or N(0, I)
        if ref_speech is None:
            z = torch.randn_like(mu_y, device=mu_y.device) / temperature   # (b, mel_dim, pmel_len)
            z_mask = y_mask  # (b, 1, pmel_len)
        else:
            z = self.decoder.forward_diffusion(ref_speech, x_mask, 0, n_timesteps)
            z_mask = y_mask
        z = mu_y + z if USE_MUY else z

        # 4. Align sequential and fixed length hidden states (to the length of the predicted phoneme-level)
        align_target_len = mu_x.shape[-1] if ADD_COND_TO_ENC else mu_y.shape[-1]
        align_target_mask = x_mask if ADD_COND_TO_ENC else y_mask
        enc_hid_cond, enc_hids_mask = self.align_combine_cond(
            psd, melstyle, spk, emo_label,
            align_target_len, align_target_mask, attn
        )

        # 6. Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(
            z.permute(0, 2, 1),
            z_mask.permute(0, 2, 1),
            mu=mu_y.permute(0, 2, 1) if USE_MUY else 0,
            n_timesteps=n_timesteps,
            stoc=stoc,
            enc_hids=enc_hid_cond.permute(0, 2, 1),
            enc_hids_mask=enc_hids_mask.permute(0, 2, 1),
        )  # (b, pmel_len, 80)
        return z[:, :, :y_max_length], decoder_outputs[:, :y_max_length, :], attn[:, :, :, :y_max_length]

    def compute_loss(self,
                     x,
                     x_lengths,
                     y,
                     y_lengths,
                     spk=None,
                     out_size=None,
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
            x (b, txt_len): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (b, ): lengths of texts in batch.
            y (b, mel_len): batch of corresponding mel-spectrograms.
            y_lengths (b, ): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk) # (b, 80, phnm_len), logw: (b, 1, phnm_len), (b, 1, phnm_len)

        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)                  # (b, phnm_len, mel_len)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const                # (b, phnm_len, mel_len)

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))  # (b, phnm_len, mel_len)
            attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # 4. Align sequential and fixed length hidden states (to the length of the predicted phoneme-level)
        align_target_len = mu_x.shape[-1] if ADD_COND_TO_ENC else log_prior.shape[-1]
        align_target_mask = x_mask if ADD_COND_TO_ENC else y_mask
        enc_hid_cond, enc_hids_mask = self.align_combine_cond(
            psd, melstyle, spk, emo_label,
            align_target_len, align_target_mask, attn
        )

        # Cut a small segment of mel-spectrogram in order to increase batch size
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
            enc_hid_cond_cut = torch.zeros(enc_hid_cond.shape[0], enc_hid_cond.shape[1], out_size,
                                   dtype=enc_hid_cond.dtype, device=enc_hid_cond.device)

            y_cut_lengths = []
            enc_hids_lengths = []
            max_x_cut_length = 1
            for i, (y_, out_offset_, enc_hid_cond_, mu_x_) in enumerate(zip(y, out_offset, enc_hid_cond, mu_x)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]

                # cut enc_hids
                cut_lower_x, cut_upper_x = get_cut_range_x(attn[i], cut_lower, cut_upper)
                x_cut_length = cut_upper_x - cut_lower_x
                if x_cut_length > max_x_cut_length:
                    max_x_cut_length = x_cut_length
                enc_hids_lengths.append(x_cut_length)
                enc_hid_cond_cut[i, :, :x_cut_length] = enc_hid_cond_[:, cut_lower_x:cut_upper_x]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            enc_hids_lengths = torch.LongTensor(enc_hids_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths, max_length=out_size).unsqueeze(1).to(y_mask)
            enc_hids_mask = sequence_mask(enc_hids_lengths, max_length=max_x_cut_length).unsqueeze(1).to(x_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask
            enc_hid_cond = enc_hid_cond_cut[:, :, :max_x_cut_length]

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # 5. Encode encoder hidden states
        diff_loss, xt = self.decoder.compute_loss(
            x0=y.permute(0, 2, 1),
            mask=y_mask.permute(0, 2, 1),
            mu=mu_y.permute(0, 2, 1) if USE_MUY else 0,
            offset=1e-5,
            enc_hids=enc_hid_cond.permute(0, 2, 1),
            enc_hids_mask=enc_hids_mask.permute(0, 2, 1)
        )
        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        return dur_loss, prior_loss, diff_loss

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
        if psd is not None:
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
        spk = self.spk_mlp(spk)
        spk_align = spk.unsqueeze(2).repeat(1, 1, align_target_len)  # (b, mel_dim, pmel_len)
        emo_label = emo_label.to(torch.float)
        emo_label_align = emo_label.unsqueeze(2).repeat(1, 1, align_target_len)

        enc_hid_cond = torch.concat([psd_aligned, spk_align, emo_label_align], dim=1)  # # (b, all_dim, phnm_len)
        enc_hids_mask = align_target_mask  # (b, )
        return enc_hid_cond, enc_hids_mask


def get_cut_range_x(attn, cut_lower, cut_upper):
    cut_lower_x = (attn[:, cut_lower] == 1).nonzero(as_tuple=False)
    cut_upper_x = (attn[:, cut_upper-1] == 1).nonzero(as_tuple=False)
    #print("cut_lower_x:", cut_lower_x)
    #print("cut_upper_x:", cut_upper_x)

    if cut_upper_x.size()[0] == 0:
        torch.set_printoptions(threshold=10_000)
        print(attn.size(), cut_lower, cut_upper)
        print(attn[:, cut_upper])
        print(attn)
    cut_lower_x = cut_lower_x[0][0]
    cut_upper_x = cut_upper_x[0][0]
    """?
    try:
    except IOError:
        print("cut_lower_x {} is empty:".format(cut_lower_x))
    try:
    except IOError:
        print("cut_upper_x {} is empty:".format(cut_upper_x))
    """
    #print("cut_lower_x, cut_upper_x:{}, {}".format(cut_lower_x, cut_upper_x))
    return cut_lower_x, cut_upper_x

def get_cut_range_x_bk(attn, cut_lower, cut_upper):
    cut_lower_x = 0
    cut_upper_x = attn.shape[0]
    start_search_upper = False
    for i in range(attn.shape[0]):
        if max(attn[i]) == 1:
            cut_lower_x = i
            start_search_upper = True
        if start_search_upper and max(attn[i+1]) == 0:
            cut_upper_x = i + 1
            break
    print("cut_lower_x, cut_upper_x:{}, {}".format(cut_lower_x, cut_upper_x))
    return cut_lower_x, cut_upper_x


