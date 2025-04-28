import math
import random

import torch

from GradTTS.model.base import BaseModule
from GradTTS.model.encoder.text_encoder import TextEncoder
from GradTTS.model.diffusion import Mish
from GradTTS.model.cond_diffusion3 import CondDiffusion

from GradTTS.model.utils import duration_loss, fix_len_compatibility, align_a2b_padcut, search_ma

from GradTTS.model.util_mask_process import (pdur2fdur, scale_dur, sequence_mask, generate_path, get_diag_from_dur2,
                                             create_p2pmask_from_seqdur, generate_diagal_fromMask)
from GradTTS.model.util_cut_speech import cutoff_inputs
from GradTTS.modules.vqpe import VQProsodyEncoder
from GradTTS.model.guided_loss import GuidedAttentionLoss
from GradTTS.utils import index_nointersperse, plot_tensor, save_plot
from GradTTS.text import symbols, extend_phone2syl
from GradTTS.model.encoder.ref_encoder import TVEncoder

ADD_COND_TO_ENC = True
USE_MUY = True

class CondGradTTSDIT3(BaseModule):
    def __init__(self,
                 n_vocab,
                 n_spks,
                 spk_emb_dim,
                 gStyle_dim,
                 gStyle_out,
                 n_feats,
                 dec_dim,
                 psd_n,
                 qk_norm=False,
                 mono_hard_mask=False,
                 mono_mas_mask=False,
                 guide_loss=False,
                 ref_encoder_type="mlp",
                 global_norm=False,
                 text_encoder_config=None,
                 ref_encoder_config=None,
                 decoder_config=None,
                 chunksize=50,
                 ):
        super(CondGradTTSDIT3, self).__init__()
        # shared parameter
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.guide_loss = guide_loss
        self.gStyle_dim = gStyle_dim

        # model options for test
        self.qk_norm = qk_norm
        self.mono_hard_mask = mono_hard_mask
        self.mono_mas_mask = mono_mas_mask
        self.phoneme_RoPE = decoder_config["stdit_config"]["phoneme_RoPE"]
        self.ref_encoder_type = ref_encoder_type
        self.global_norm = global_norm
        self.chunksize = chunksize

        # fine-ad

        # Condition embed
        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        #self.emo_emb = torch.nn.Embedding(5, emo_emb_dim)
        self.spk_mlp = torch.nn.Sequential(torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4), Mish(),
                                           torch.nn.Linear(spk_emb_dim * 4, spk_emb_dim))
        self.psd_mlp = torch.nn.Sequential(torch.nn.Linear(psd_n, psd_n * 4), Mish(),
                                           torch.nn.Linear(psd_n * 4, spk_emb_dim))
        if gStyle_dim > 0:
            self.emo_mlp = torch.nn.Sequential(torch.nn.Linear(gStyle_dim, gStyle_dim), Mish(),
                                               torch.nn.Linear(gStyle_dim, gStyle_out))  # ?? 128 is too large ??
        if self.ref_encoder_type == "vae":
            self.ref_encoder = VQProsodyEncoder(**ref_encoder_config)
        elif self.ref_encoder_type == "vaeEma":
            self.ref_encoder = TVEncoder(**ref_encoder_config)
        else:
            lStyle_dim, lStyle_out = ref_encoder_config["lStyle_dim"], ref_encoder_config["lStyle_out"]
            self.ref_encoder = torch.nn.Sequential(torch.nn.Linear(lStyle_dim, lStyle_dim), Mish(),
                                                    torch.nn.Linear(lStyle_dim, lStyle_out))

        text_encoder_config_ext = dict(n_feats=n_feats, n_spks=n_spks, spk_emb_dim=spk_emb_dim,
                                       n_vocab=n_vocab, **text_encoder_config)
        decoder_config_ext = dict(n_feats=n_feats, n_spks=n_spks,
                                  dim=dec_dim, gStyle_dim=gStyle_out, **decoder_config)

        # Encoder
        self.encoder = TextEncoder(**text_encoder_config_ext)

        # Decoder
        print(decoder_config_ext)
        self.decoder = CondDiffusion(**decoder_config_ext)
        if self.guide_loss:
            self.guided_attn = GuidedAttentionLoss(sigma=0.1, alpha=1.0)

    def predict_prior(self,
                      x,
                      x_lengths,
                      spk,
                      length_scale=1.0
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
                seed=1,
                syl_start=None,
                enh_ind_syn_ref = None,
                ):
        """

        Args:
            x:
            x_lengths:
            n_timesteps:
            temperature:
            stoc:
            spk:
            length_scale:
            psd:
            emo_label:
            melstyle:
            melstyle_lengths:
            guidence_strength:
            seed:
        Returns:

        """
        torch.manual_seed(seed)
        x, x_lengths = self.relocate_input([x, x_lengths])

        # 1. Embed condition
        spk_emb = self.spk_mlp(self.spk_emb(spk))
        if self.gStyle_dim > 0:
            emo_label = torch.nn.functional.one_hot(emo_label, num_classes=self.gStyle_dim).to(torch.float)
            emo_label = self.emo_mlp(emo_label)
        if self.ref_encoder_type == "vae":  # melstyle = mel
            melstyle, _, _, _ = self.ref_encoder(melstyle.transpose(1, 2))  # (b, mel_len, dict_dim)
        elif self.ref_encoder_type == "vaeEma":
            melstyle_mask = torch.unsqueeze(sequence_mask(melstyle_lengths, melstyle.size(2)), 1).to(x.dtype)  # (b, 1, mel_len)
            melstyle_beforeVQ, melstyle_lengths, _ = self.ref_encoder(melstyle, melstyle_mask)  # IN/OUT: (b, dict_dim, mel_len)
        else:
            melstyle = self.ref_encoder(melstyle.transpose(1, 2)).transpose(1, 2)   # (b, mel_len, 80)

        # 2. Get prior and p-level duration
        mu_y, y_mask, y_max_length, attn, attn_mask = self.predict_prior(x, x_lengths, spk_emb, length_scale)
        encoder_outputs = mu_y[:, :, :y_max_length]
        p_dur = torch.sum(attn, -1).squeeze(1)  # (b, 1, mu_x_len, mu_y_len) -> (b, mu_x_len)
        k_dur = scale_dur(p_dur, melstyle_lengths)  # (b, 1, melstyle_len)
        #syl_start_scale = scale_dur(syl_start, melstyle_lengths)

        if self.phoneme_RoPE == "phone" or self.phoneme_RoPE == "sel":
            syl_start = syl_start if self.phoneme_RoPE == "sel" else None
            q_seq_dur = pdur2fdur(p_dur, y_mask.squeeze(1), syl_start)  # (b, 1, mu_x_len) -> (b, 1, mu_y_len)
            k_seq_dur = pdur2fdur(k_dur, sequence_mask(melstyle_lengths).long(), syl_start) # ???Only used for parallel data (unpara need extra input k_dur)
        else:
            q_seq_dur = None
            k_seq_dur = None

        # 3. Get sample terminal distribution mu_y + N(mu_y, I)
        variance = torch.randn_like(mu_y, device=mu_y.device) / temperature
        z = mu_y + variance

        # 4. gather attn_mask and enhance input
        attnCross_mask = y_mask.repeat(1, melstyle.shape[2], 1).transpose(1, 2)  # !! strange ||
        # get enhancing phone duration given index
        if enh_ind_syn_ref is not None:
            q_enh_phone_indx, k_enh_phone_indx = enh_ind_syn_ref
            refenh_ind_dur, synenh_ind_dur = ((q_enh_phone_indx, p_dur[:, q_enh_phone_indx]),
                                              (k_enh_phone_indx, k_dur[:, k_enh_phone_indx]))
        else:
            refenh_ind_dur, synenh_ind_dur = None, None

        ################ check code
        #guide_matrix = create_p2pmask_from_seqdur(q_seq_dur, k_seq_dur)  # (b, mu_y_len, melstyle_len)
        #save_plot(guide_matrix[0].cpu(), f"pguide_attnmask_epoch_test.png")

        # 5. synthesize speeehcc
        decoder_outputs, self_attns_list, cross_attns_list = self.decoder(
            z,
            y_mask,
            mu_y,
            n_timesteps,
            stoc=stoc,
            melstyle=melstyle,
            emo_label=emo_label,
            attn_mask=attnCross_mask,
            q_seq_dur=q_seq_dur,
            k_seq_dur=k_seq_dur,
            refenh_ind_dur=refenh_ind_dur,
            synenh_ind_dur=synenh_ind_dur
        )

        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        return (encoder_outputs, decoder_outputs, attn[:, :, :y_max_length], self_attns_list, cross_attns_list,
                p_dur, k_dur)
    
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
                     emo_label=None,
                     syl_start=None,
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

        # 1. Embed condition
        spk_emb = self.spk_mlp(self.spk_emb(spk))
        melstyle_mask = torch.unsqueeze(sequence_mask(melstyle_lengths, melstyle.size(2)), 1).to(x.dtype)
        if self.gStyle_dim > 0:
            emo_label = torch.nn.functional.one_hot(emo_label, num_classes=self.gStyle_dim).to(torch.float)
            emo_label = self.emo_mlp(emo_label)
        if self.ref_encoder_type == "vae":  # melstyle = mel
            melstyle, commit_loss, vq_loss, codes = self.ref_encoder(melstyle.transpose(1, 2))  # (b, mel_len, dict_dim)
        elif self.ref_encoder_type == "vaeEma":
            #melstyle_mask = torch.unsqueeze(sequence_mask(melstyle_lengths, melstyle.size(2)), 1).to(x.dtype)  # (b, 1, mel_len)
            melstyle_beforeVQ, melstyle, vq_loss = self.ref_encoder(melstyle, melstyle_mask)  # IN/OUT: (b, dict_dim, mel_len)
            commit_loss = torch.tensor(0.0, dtype=torch.float32)
        else:
            melstyle = self.ref_encoder(melstyle.transpose(1, 2)).transpose(1, 2)   # (b, mel_len, 80)

        # 2. Duration related
        ## Get predict mu_x (encoded x) and logw (log-scaled token durations)
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk_emb)
        y_max_length = y.shape[-1]
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        ## Get gd attn, logw_ by MAS
        with torch.no_grad():
            attn = search_ma(mu_x, y, attn_mask, xy_dim=self.n_feats)  # attn value = 1
            attn = attn.detach()
        ## Duration loss between logw and logw_
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        ## convert gd attn to gd dur, seq_dur (ONLY for p/s-level embedding)
        if self.phoneme_RoPE == "phone" or self.phoneme_RoPE == "sel":
            p_dur = torch.sum(attn, -1)             # (b, mu_x_len, mu_y_len) -> (b, mu_x_len)
            syl_start = syl_start if self.phoneme_RoPE == "sel" else None

            k_dur = scale_dur(p_dur, melstyle_lengths)
            #print("phone_num of p_dur, k_dur", torch.count_nonzero(p_dur, dim=1), torch.count_nonzero(k_dur, dim=1))
            #print("p_dur[:4], k_dur[:4]", p_dur[:4, :], k_dur[:4, :])
            #print("y_len[:4], mel_style_len[:4]", y_lengths[:4], melstyle_lengths[:4])

            q_seq_dur = pdur2fdur(p_dur, y_mask.squeeze(1), syl_start).unsqueeze(1)     # (b, mu_x_len) -> (b, 1, mu_y_len)
            k_seq_dur = pdur2fdur(k_dur, sequence_mask(melstyle_lengths).long(), syl_start).unsqueeze(1) # (b, 1, melstyle_len)
            #print("phone_num of q_seq_dur, k_seq_dur", torch.max(q_seq_dur[:4, 0, :], dim=1)[0], torch.max(k_seq_dur[:4, 0, :], dim=1)[0])
            #print("q_seq_dur, k_seq_dur", q_seq_dur[:4, 0, :], k_seq_dur[:4, 0, :])
        else:
            q_seq_dur = None
            k_seq_dur = None

        # 3. Cut a small segment of mel-spectrogram (to increase batch size)
        y, y_lengths, y_mask, melstyle_cond, melstyle_lengths, attn, q_seq_dur, k_seq_dur = cutoff_inputs(
            y, y_lengths, y_mask, melstyle, melstyle_lengths, attn, out_size, y_dim=self.n_feats, q_seq_dur=q_seq_dur, k_seq_dur=k_seq_dur) # melstyle.shape[2] = y.shape[2]
        #print("phone_num of q_seq_dur, k_seq_dur", torch.max(q_seq_dur[:4, 0, :], dim=1)[0], torch.max(k_seq_dur[:4, 0, :], dim=1)[0])
        #print("q_seq_dur, k_seq_dur", q_seq_dur[:4, 0, :], k_seq_dur[:4, 0, :])

        if q_seq_dur is not None:
            q_seq_dur = q_seq_dur.squeeze(1)
            k_seq_dur = k_seq_dur.squeeze(1)

        # 4. Gather attn_mask, mu_y for model input
        ## Create cross_attn mask
        attnCross_mask = y_mask.repeat(1, melstyle_cond.shape[2], 1).transpose(1, 2)  # ???
        ## Get mu_y by attn
        save_plot(attn[1].detach().cpu(), "muxy_attn.png")
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)) # attn: (b, mu_x_len, mu_y_len)
        mu_y = mu_y.transpose(1, 2)

        # 5. Compute loss
        ## Difussion loss
        #print("mu_y", mu_y[0])
        #print("melstyle_cond", melstyle_cond[0])
        diff_loss, xt, attn_selfs, attn_crosses = self.decoder.compute_loss(
            y,       # (b, mel_dim, cut_l)
            y_mask,  # (b, )
            mu_y,    # (b, mel_dim, cut_l)
            offset=1e-5,
            melstyle=melstyle_cond,  # (b, mel_dim, cut_l)
            emo_label=emo_label,     # (b, )
            attnCross=attnCross_mask,
            q_seq_dur=q_seq_dur,   # (b, cut_l),
            k_seq_dur=k_seq_dur
        )   # attn_maps: ((b, h, l_q * d_q, l_k), (b, h, l_q * d_q, l_k))
        ## Prior loss
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        ## Monotonic attention loss
        blk, batch, head, ql, kl = attn_crosses.shape
        if self.guide_loss == "psuedo":    # psuedo, p2p, s2s
            guide_matrix, _ = self.guided_attn.create_guide_mask(
                ilens=y_lengths, olens=melstyle_lengths, chunksize=self.chunksize) # (b, ilens_max, olens_max)
            guide_matrix = guide_matrix.unsqueeze(1).unsqueeze(0)
            guide_matrix = guide_matrix.repeat(blk, 1, head, 1, 1)
            monAttn_loss = torch.mean(attn_crosses * guide_matrix)
            #save_plot(attn_crosses[0, 1, 0].detach().cpu(), "attn_crosses_cutsize.png")
            #save_plot(guide_matrix[0, 1, 0].detach().cpu(), "guide_matrix_cutsize.png")
        elif self.guide_loss == "p2p":
            guide_matrix = create_p2pmask_from_seqdur(q_seq_dur, k_seq_dur)  #  (b, mu_y_len, melstyle_len)
            #save_plot(guide_matrix[0].cpu(), f"pguide_attnmask_epoch.png")
            #print(guide_matrix[0])
            guide_matrix = guide_matrix.unsqueeze(1).unsqueeze(0).repeat(blk, 1, head, 1, 1)
            cross_attn_mask = attnCross_mask.unsqueeze(1).unsqueeze(0).repeat(blk, 1, head, 1, 1)
            #print(attn_crosses.shape, guide_matrix.shape, cross_attn_mask.shape)
            #monAttn_loss = torch.mean(attn_crosses * guide_matrix * cross_attn_mask)
            monAttn_loss = torch.mean(attn_crosses * guide_matrix)
        else:
            monAttn_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)

        ## VAE loss
        if self.ref_encoder_type == "vae" or self.ref_encoder_type == "vaeEma":
            return dur_loss, prior_loss, diff_loss, monAttn_loss, commit_loss, vq_loss
        else:
            commit_loss, vq_loss = (torch.tensor(0.0, dtype=torch.float32, requires_grad=False),
                                    torch.tensor(0.0, dtype=torch.float32, requires_grad=False))
            return dur_loss, prior_loss, diff_loss, monAttn_loss, commit_loss, vq_loss


def get_attnCross_mask(attnXy, attnPs=None, p_style_dur=None):
    """
    Args:
        attnXy:            (b, p_l, muy_l)
        attnPs:            (b, p_l, style_l)
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