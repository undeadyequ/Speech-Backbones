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
from GradTTS.model.base import BaseModule
from GradTTS.model.encoder.text_encoder import TextEncoder
from GradTTS.model.diffusion import Mish
from GradTTS.model.cond_diffusion2 import CondDiffusion
from GradTTS.model.utils import (duration_loss,
                                 fix_len_compatibility, align, align_a2b_padcut, cut_pad_start_end)
from GradTTS.modules.vqpe import VQProsodyEncoder
from GradTTS.model.guided_loss import GuidedAttentionLoss
from GradTTS.model.utils import search_ma
from GradTTS.utils import index_nointersperse
from GradTTS.text import symbols
from GradTTS.model.encoder.ref_encoder import TVEncoder
from GradTTS.model.util_mask_process import sequence_mask, generate_path, generate_diagal_fromMask, get_diag_from_dur2

ADD_COND_TO_ENC = True
USE_MUY = True


class CondGradTTSDIT(BaseModule):
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
                 sample_channel_n, 
                 beta_min, 
                 beta_max, 
                 pe_scale, 
                 unet_type,
                 att_type, 
                 att_dim, 
                 heads, 
                 p_uncond,
                 psd_n,
                 melstyle_n,  # 768
                 diff_model="STDitMocha",
                 ref_encoder="mlp",
                 guided_attn=False,
                 dit_mocha_config=None,
                 stdit_config=None,
                 stditMocha_config=None,
                 tvencoder_config=None,
                 vqvae=None,
                 ):
        super(CondGradTTSDIT, self).__init__()
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
        self.guided_attn = guided_attn

        self.estimator_type, self.ref_encode_type = diff_model, ref_encoder
        
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
                                     att_dim, heads, p_uncond,
                                     self.estimator_type,
                                     dit_mocha_config=dit_mocha_config,
                                     stditMocha_config=stditMocha_config,
                                     stdit_config=stdit_config)

        # mlp, vae, psdMlp
        if self.ref_encode_type == "vae":
            self.vqpe = VQProsodyEncoder(**vqvae)
        elif self.ref_encode_type == "vaeEma":
            self.tv_encoder = TVEncoder(**tvencoder_config)
        else:
            self.melstyle_mlp = torch.nn.Sequential(torch.nn.Linear(melstyle_n, melstyle_n), Mish(),
                                                    torch.nn.Linear(melstyle_n, n_feats))

        if self.guided_attn:
            self.guided_attn = GuidedAttentionLoss(sigma=0.4, alpha=1.0)

        self.monotonic_approach = stditMocha_config["monotonic_approach"]

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
                melstyle_lengths=None,
                guidence_strength=3.0,
                attn_hardMask=None,
                seed=1
                ):
        torch.manual_seed(seed)
        x, x_lengths = self.relocate_input([x, x_lengths])

        # Embed condition
        spk_emb = self.spk_mlp(self.spk_emb(spk))
        if self.ref_encode_type == "vae":  # melstyle = mel
            melstyle_t = melstyle.transpose(1, 2)
            melstyle, _, _, _ = self.vqpe(melstyle_t)  # (b, mel_len, dict_dim)
            #melstyle = self.melstyle_mlp(melstyle_zq)  # temparally
        elif self.ref_encode_type == "vaeEma":
            melstyle_mask = torch.unsqueeze(sequence_mask(melstyle_lengths, melstyle.size(2)), 1).to(x.dtype)  # (b, 1, mel_len)
            melstyle_beforeVQ, melstyle_lengths, _ = self.tv_encoder(melstyle, melstyle_mask)  # IN/OUT: (b, dict_dim, mel_len)
        else:
            if melstyle.shape[1] == 3:
                p_style_dur = melstyle[:, 2, :]
                melstyle = melstyle[:, :2, :]
            melstyle = self.melstyle_mlp(melstyle.transpose(1, 2)).transpose(1, 2)   # (b, mel_len, 80)

        # get prior
        mu_y, y_mask, y_max_length, attn, attn_mask = self.predict_prior(x, x_lengths, spk_emb, length_scale)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Get sample latent representation from terminal distribution N(mu_y, I)
        variance = torch.randn_like(mu_y, device=mu_y.device) / temperature
        z = mu_y + variance

        # Generate sample by performing reverse dynamics
        #melstyle = melstyle.transpose(1, 2)

        # create attnCross_mask (monotoic attention mask)

        if self.estimator_type == "STDitMocha":
            if self.monotonic_approach == "hard_phoneme":
                attnCross_mask = get_attnCross_mask(attnXy=attn.squeeze(1), p_style_dur=p_style_dur)
                #attnCross_mask = None
            elif self.monotonic_approach == "hard":
                attn_mask_cross = y_mask.repeat(1, 1, melstyle.shape[2]).transpose(1, 2)         # (b, tq, ts) <- ts is not masked
                # print("before diag:", attn_mask_cross)
                #attnCross_mask = generate_diagal_fromMask(attn_mask_cross)
                attnCross_mask = None
            elif self.monotonic_approach == "guidloss" or self.monotonic_approach == "mocha":
                attnCross_mask = None  # (b, tq, ts)
            elif self.monotonic_approach == "mas":
                attnCross_mask = search_ma(mu_y, melstyle, y_mask)
            else:
                attnCross_mask = y_mask.repeat(1, melstyle.shape[2], 1).transpose(1, 2)
        else:
            attnCross_mask = None

        decoder_outputs, self_attns_list, cross_attns_list = self.decoder(z,
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
                                       attn_mask=attnCross_mask)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length], self_attns_list, cross_attns_list
    
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
                     melstyle_lengths=None,
                     emo_label=None
                     ):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
        Args:
            x (b, p_l):  batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (b,): lengths of texts in batch.
            y (b, mel_dim, mel_l): batch of corresponding mel-spectrograms.
            y_lengths (b, ): lengths of mel-spectrograms in batch.
            out_size (int): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
            melstyle (b, ls_dim, ls_l)
            emo_label (b,)

        attn_mask: (b, p_l, m_l)                # xy_attn
        attn_cross_mask: (b)
        attn_cross_diag_mask: (b, m_l, s_l)     # ystyle_attn

        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        # Melstyle embedding
        if self.ref_encode_type == "vae":  # melstyle = mel
            melstyle_t = melstyle.transpose(1, 2)
            melstyle, commit_loss, vq_loss, codes = self.vqpe(melstyle_t)  # (b, mel_len, dict_dim)

        elif self.ref_encode_type == "vaeEma":
            melstyle_mask = torch.unsqueeze(sequence_mask(melstyle_lengths, melstyle.size(2)), 1).to(x.dtype)  # IN/OUT: (b, dict_dim, mel_len)
            melstyle_beforeVQ, melstyle, vq_loss = self.tv_encoder(melstyle, melstyle_mask)  # (b, mel_len, dict_dim)
            commit_loss = torch.tensor(0.0, dtype=torch.float32)
        else:
            if melstyle.shape[1] == 3:
                p_style_dur = melstyle[:, 2, :].unsqueeze(1)
                melstyle = melstyle[:, :2, :]
            melstyle = self.melstyle_mlp(melstyle.transpose(1, 2)).transpose(1, 2)   # (b, mel_len, 80)
            if melstyle.shape[1] == 3:
                melstyle = torch.concatenate([melstyle, p_style_dur], dim=1)  # add PSD to cutoff by once

        # Get mu_x (encoded x) and logw (log-scaled token durations)
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use gd logw_ by attn which predicted by MAS (find most likely alignment `attn` between text and mel-spectrogram)
        with torch.no_grad():
            attn = search_ma(mu_x, y, attn_mask, xy_dim=self.n_feats)  # attn value = 1
            attn = attn.detach()

        # Duration loss between logw and logw_
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Align melstyle to mu_y for cut
        if melstyle.shape[2] != y.shape[2]:
            #print("Align melstyle to y length!")
            melstyle = align_a2b_padcut(melstyle, y.shape[-1], pad_value=0, pad_type="right")   # melstyle.shape[2] = y.shape[2]

        # Cut a small segment of mel-spectrogram (to increase batch size)
        y, y_lengths, y_mask, melstyle_cond, attn = cutoff_inputs(y, y_lengths, y_mask, melstyle, attn, out_size,
                                                             y_dim=self.n_feats) # melstyle.shape[2] = y.shape[2]
        # Create cross_attn mask
        if self.monotonic_approach == "hard_phoneme":
            p_style_dur = melstyle_cond[:, -1, :]     # (b, s_l), 0001111, s_l=m_l
            melstyle_cond = melstyle_cond[:, :-1, :]  # (b, D, m_l)

        # create attnCross_mask (monotoic attention mask)
        if self.monotonic_approach == "STDitMocha":
            attnCross_mask = get_attnCross_mask(attnXy=attn, p_style_dur=p_style_dur)
        else:
            attnCross_mask = None

        # Get mu_y by attn
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)) # attn: (b, mu_x_len, mu_y_len)
        mu_y = mu_y.transpose(1, 2)

        # Difussion loss
        diff_loss, xt, attn_selfs, attn_crosses = self.decoder.compute_loss(
            y,       # (b, mel_dim, cut_l)
            y_mask,  # (b, )
            mu_y,    # (b, mel_dim, cut_l)
            offset=1e-5,
            spk=spk, # (b, )
            psd=psd, # (b, )
            melstyle=melstyle_cond,  # (b, mel_dim, cut_l)
            emo_label=emo_label,     # (b, )
            align_len=mu_y.shape[-1],
            attnCross=attnCross_mask
        )   # attn_maps: ((b, h, l_q * d_q, l_k), (b, h, l_q * d_q, l_k))
        
        # Prior loss
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        
        # Monotonic attention loss
        ## Convert attn_mux_muy to attn_muy_style
        
        if self.guided_attn:
            # Semi-ground truth: rough digonal
            monAttn_loss = 0
            blk, batch, head, ql, kl = attn_crosses.shape
            for attn_cross in attn_crosses:
                ilens = y_lengths
                #olens = y_lengths * 80 / 2 ** 2  # patchsize = 2
                olens = y_lengths
                for h in range(head):
                    monAttn_loss += self.guided_attn(attn_cross[:, h, :, :], olens, ilens, attn_cross.shape[-2:])
            """Ground Truth
            attn_muy_style = torch.rand_like(attn)
            assert attn_muy_style.size() == attn_maps.size()   # how ?
            mon_attention_mask = create_mon_attn_mask(attn_uy=attn, attn_ref=attn_muy_style)
            monAttn_loss = mon_attention_loss(attn_maps, mon_attention_mask)
            """
        else:
            monAttn_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
        # VAE loss
        if self.ref_encode_type == "vae" or self.ref_encode_type == "vaeEma":
            return dur_loss, prior_loss, diff_loss, monAttn_loss, commit_loss, vq_loss
        else:
            commit_loss, vq_loss = (torch.tensor(0.0, dtype=torch.float32, requires_grad=False),
                                    torch.tensor(0.0, dtype=torch.float32, requires_grad=False))
            return dur_loss, prior_loss, diff_loss, monAttn_loss, commit_loss, vq_loss

def get_attnCross_mask(attnXy, attnPs=None, p_style_dur=None):
    """

    Args:
        attnXy:
        attnPs:
        p_style_dur:

    Returns:

    """

    if attnPs is None:
        # x_no_spaceIndex = index_nointersperse(x, item=lensymbol) # .shape[0]: <x.shape[0]
        # attnXy_spacerm = attn[:, x_no_spaceIndex, :]             # attn: (b, pp_l, m_l) -> (b, p_l, m_l)

        attnXy_spacerm = attnXy  ## (b, p_l, mcut_l)
        attnPs = get_diag_from_dur2(p_style_dur)  ## (b, p_l, scut_l)   # !s_l == m_l

        ############CHECK1############
        if attnXy_spacerm.shape[1] != attnPs.shape[1]:
            # print("attnxy {}, attnps {} shape should be same!".format(attnXy_spacerm.shape, attnPs.shape))
            attnPs = align_a2b_padcut(attnPs.transpose(1, 2), attnXy_spacerm.shape[1],
                                      pad_value=0, pad_type="right").transpose(1, 2)
        assert attnXy_spacerm.shape[1] == attnPs.shape[1]
        attnYs_space_rm = torch.matmul(attnXy_spacerm.transpose(1, 2), attnPs)  # (b, m_l, s_l)

        ############ CHECK23 ############
        # from GradTTS.utils import save_plot
        # print("attn_mask_cross -> block diag:", attn_mask_cross)
        # print("attnXy_spacerm and attnPs should be diagnal with 1", attnXy_spacerm, attnPs)
        # save_plot(attnXy_spacerm[0].cpu(), f'watchImg/attnXy_spacerm.png')
        # save_plot(attnPs[0].cpu(), f'watchImg/attnPs.png')
        # save_plot(attnYs_space_rm[0].cpu(), f'watchImg/attnYs_space_rm.png')  # (b, m_l, s_l)
    else:
        attnYs_space_rm = torch.matmul(attnXy.transpose(1, 2), attnPs)  # (b, m_l, s_l)
    return attnYs_space_rm


def cutoff_inputs(y, y_lengths, y_mask, melstyle, attn, cut_size, y_dim=80):
    """
    cut off inputs for larger batches
    Returns:
    """
    if not isinstance(cut_size, type(None)):
        # randomly select start and end point
        max_offset = (y_lengths - cut_size).clamp(0)
        offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
        out_offset = torch.LongTensor([
            torch.tensor(random.choice(range(start, end)) if end > start else 0)
            for start, end in offset_ranges
        ]).to(y_lengths)

        # initiate tensor
        attn_cut = torch.zeros(attn.shape[0], attn.shape[1], cut_size, dtype=attn.dtype, device=attn.device)
        y_cut = torch.zeros(y.shape[0], y_dim, cut_size, dtype=y.dtype, device=y.device)
        melstyle_cut = torch.zeros(melstyle.shape[0], melstyle.shape[1], cut_size,
                                        dtype=melstyle.dtype, device=melstyle.device)
        y_cut_lengths = []
        melstyle_lengths = []

        # assign values
        for i, (y_, out_offset_, enc_hid_cond_) in enumerate(zip(y, out_offset, melstyle)):
            y_cut_length = cut_size + (y_lengths[i] - cut_size).clamp(None, 0)
            y_cut_lengths.append(y_cut_length)
            cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
            y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
            attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            melstyle_lengths.append(y_cut_length)
            melstyle_cut[i, :, :y_cut_length] = enc_hid_cond_[:, cut_lower:cut_upper]

        y_cut_lengths = torch.LongTensor(y_cut_lengths)
        y_cut_mask = sequence_mask(y_cut_lengths, max_length=cut_size).unsqueeze(1).to(
            y_mask)  # Set length to cut_size to enable fix length learning
        melstyle_lengths = torch.LongTensor(melstyle_lengths)

        attn = attn_cut
        y = y_cut
        y_mask = y_cut_mask
        melstyle = melstyle_cut
        y_lengths = y_cut_lengths

    return y, y_lengths, y_mask, melstyle, attn


def get_cut_range_x(attn, cut_lower, cut_upper):
    cut_lower_x = (attn[:, cut_lower] == 1).nonzero(as_tuple=False)
    cut_upper_x = (attn[:, cut_upper-1] == 1).nonzero(as_tuple=False)
    if cut_upper_x.size()[0] == 0:
        torch.set_printoptions(threshold=10_000)
        #print(attn.size(), cut_lower, cut_upper)
        #print(attn[:, cut_upper])
        #print(attn)
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