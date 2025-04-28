import os.path
from einops import rearrange
import math
import torch
import torch.nn.functional as F
import numpy as np
import librosa
from GradTTS.model.utils import extract_pitch
from GradTTS.model.base import BaseModule
from GradTTS.model.diffusion import *
from GradTTS.model.utils import align, align_a2b_padcut
import matplotlib.pyplot as plt
from bisect import bisect
from pathlib import Path
from GradTTS.model.util_maskCreation import create_left_right_mask, create_pitch_bin_mask

class GradLogPEstimator2dCond(BaseModule):
    def __init__(self,
                 dim,
                 sample_channel_n=2,
                 dim_mults=(1, 2, 4),
                 groups=8,
                 n_spks=None,
                 spk_emb_dim=64,
                 n_feats=80,
                 pe_scale=1000,
                 emo_emb_dim=768,
                 att_type="linear",
                 attn_gran="frame2bin",
                 att_dim=128,
                 heads=4,
                 p_uncond=0.9,
                 ):
        """
        classifier free guidance
        Args:
            dim:
            dim_mults:
            groups:
            n_spks:
            spk_emb_dim:
            n_feats:
            pe_scale:
            emo_emb_dim:
            att_type: "linear" for linearAttention where input are simply concatenated,
                        "cross" for crossAttention
        """
        super(GradLogPEstimator2dCond, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.n_spks = n_spks if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale
        self.att_type = att_type
        self.heads = heads
        self.p_uncond = p_uncond

        if n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4), Mish(),
                                               torch.nn.Linear(spk_emb_dim * 4, n_feats))
        self.emo_mlp = torch.nn.Sequential(torch.nn.Linear(emo_emb_dim, emo_emb_dim), Mish(),
                                           torch.nn.Linear(emo_emb_dim, n_feats))
        self.psd_mlp = torch.nn.Sequential(torch.nn.Linear(3, 16), Mish(),
                                           torch.nn.Linear(16, n_feats))
        self.enc_hid_dim = n_feats * 3

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.t_mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(),
                                         torch.nn.Linear(dim * 4, dim))
        # Sample channel
        total_dim = sample_channel_n + (1 if n_spks > 1 else 0)  # 4 include spk, emo, psd, x
        # Enc_hid channel
        dims = [total_dim, *map(lambda m: dim * m, dim_mults)] # [3, 64, 128, 256]
        in_out = list(zip(dims[:-1], dims[1:]))  # [(3, 64), (64, 128), (128, 256)]
        #print(in_out)
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        self.num_resolutions = len(in_out)

        # Set unet
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (self.num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                # MultiAttention3->frame2frame; MultiAttention2->bin2frame (simFrame2fame?)
                #Residual(Rezero(LinearAttention(dim_out) if att_type == "linear" else
                #                MultiAttention(dim_out, self.enc_hid_dim, att_dim, heads, dim))),
                Residual(Rezero(LinearAttention(dim_out) if att_type == "linear" else
                                MultiAttention2(dim_out, self.enc_hid_dim, att_dim, heads))),

                Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        #self.mid_attn = Residual(Rezero(LinearAttention(mid_dim) if att_type == "linear" else
        #                                MultiAttention(mid_dim, self.enc_hid_dim, att_dim, heads)))
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim) if att_type == "linear" else
                                        MultiAttention2(mid_dim, self.enc_hid_dim, att_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                #Residual(Rezero(LinearAttention(dim_in) if att_type == "linear" else
                #                MultiAttention(dim_in, self.enc_hid_dim, att_dim, heads, dim / 2 if is_last else dim))),
                Residual(Rezero(LinearAttention(dim_in) if att_type == "linear" else
                            MultiAttention2(dim_in, self.enc_hid_dim, att_dim, heads))),
                Upsample(dim_in)]))
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self,
                x,
                mask,
                mu,
                t,
                spk=None,
                psd=None,
                melstyle=None,
                emo_label=None,
                align_len=None,
                align_mtx=None,
                guidence_strength=3.0,
                return_attmap=False,
                attn_mask=None,
                attn_img_time_ind=(0, 10, 20, 30, 40, 49)
                ):
        """
        Predict noise on classifier-free guidance where the noised of conditioned and unconditioned are weighted added.

        att_type option
            linear (self-attention):
                x = x + emo + psd
            CrossAtt (Not used):
                q = x
                k, v = emo + psd ? (No text information)

        Args:
            x:   (b, 80, mel_len)   # mel_len = 100 (fixed)
            mask:(b, 1, mel_len)
            mu:  (b, 80, mel_len)
            t:   (b, 64)
            spk: (b, spk_emb)
            emo_label: (b, emo_num)
            psd [tuple]: ((b, word_len), (b, word_len), (b, word_len))
            melstyle: (b, 80, mel_len)
        Returns:
            output: (b, 80, mel_len)
        """
        # Regularize unet input1: xt, enc_hids, and attn_mask (The ref len should = mu_y len)
        x, hids = align_cond_input(x, mu, spk, melstyle, emo_label, align_len, align_mtx, self.att_type)  # (b, 3, 80, mel_len)
        if attn_mask is not None:
            attn_mask = align_a2b_padcut(attn_mask, align_len)


        # Joint training diff model with uncond/cond by p_uncondition
        # ??2. Expected same result with different input makes training impossible.
        if self.training:
            p = torch.rand(1)
            if p < self.p_uncond:
                b, c, d_q, l_q = x.shape
                hids = x.clone()   # <- discard conditioning with p_uncond
                hids = hids.view(b, -1, l_q)
            res, attn_amp = self.forward_to_unet(x, mask, hids, t, return_attmap=return_attmap)
        # Conditional sampling with weighted noise of uncond/cond
        else:
            if return_attmap:
                cond_res, attn_amp = self.forward_to_unet(x,
                                                          mask,
                                                          hids,
                                                          t,
                                                          return_attmap=return_attmap,
                                                          attn_mask=attn_mask,
                                                          attn_img_time_ind=attn_img_time_ind
                                                          )
            else:
                cond_res = self.forward_to_unet(x,
                                                mask,
                                                hids,
                                                t,
                                                return_attmap=return_attmap,
                                                attn_mask=attn_mask,
                                                attn_img_time_ind=attn_img_time_ind
                                                )
            # Estimated noise with weighted cond_res and noncond_res
            if self.p_uncond != 0.0:
                b, c, d_q, l_q = x.shape
                hids = x.clone()  # <- NO condition
                hids = hids.view(b, -1, l_q)
                uncond_res = self.forward_to_unet(x, mask, hids, t, attn_mask=attn_mask,
                                                  attn_img_time_ind=attn_img_time_ind)
                res = (1 + guidence_strength) * cond_res - guidence_strength * uncond_res
            # Estimated cond_res and noncond_res
            else:
                res = cond_res

            if return_attmap:
                return res, attn_amp
        return res, attn_amp

    def forward_to_unet(self,
                        x,
                        mask,
                        hids,
                        t,
                        show_attmap=True,
                        return_attmap=False,
                        attn_mask=None,
                        attn_img_time_ind=(0, 10, 20, 30, 40, 49)
                        ):
        """
        Forward to unet model with single hids <- modify to unet_2d_cond_speech class in v2

        1. show attention map
        - Save p2f_map_score during inference (cut/pad?)
            - Head  <- all head
            - Layer <- for first and last layer
            - Time  <- [0, 10, 20, 30, 40, 49]
        - get p_dur, phonemes by MFA from speech, text
        - get p2p_map_score by attention_map_score function.
        - visualize p2p_map_score from p2p_map_score and p_dur

        attn_map: att(b, h, l_q * d_q, l_k)   # show all head?

        Returns:

        """
        input = torch.clone(x)
        t0 = t

        # setup for return_attn
        #attn_img_time_ind = [0, 10, 20, 30, 40, 49]  # <- check front (near 1) or back (near 0)
        n_timesteps = 50
        attn_img_time = torch.tensor(
            [(1.0 - (ind + 0.5) * (1.0 / n_timesteps)) for ind in attn_img_time_ind], dtype=t.dtype, device=t.device
        )
        return_attn = []

        # Regularize Unet input2: Time projection
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.t_mlp(t)
        hiddens = []

        # mask
        mask = mask.unsqueeze(1)
        masks = [mask]
        attn_masks = [attn_mask]

        for i, (resnet1, resnet2, attn, downsample) in enumerate(self.downs):
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)   # (b, c, d, l) -> (b, c_out_i, d, l)
            x = resnet2(x, mask_down, t)   # -> (b, c_out_i, d, l)
            x, attn_map = attn(x) if self.att_type == "linear" else attn(x, key=hids, value=hids,
                                                                         attn_mask=attn_mask)  # -> (b, c_out_i, d, l), (b, h, l_q * d_q, l_k)
            hiddens.append(x)
            x = downsample(x * mask_down)  # -> (b, c_out_i, d/2, l/2)
            masks.append(mask_down[:, :, :, ::2])
            if attn_mask is not None:
                attn_mask = attn_mask[:, :, ::4, :]  # Q: d/2 * l/2 = d*l*1/4   KV: No changed
            attn_masks.append(attn_mask)

            # Save p2f_map_score of 1st layer
            #if i == 0 and t0[0] in attn_img_time and show_attmap and not self.training:
            if show_attmap:
                return_attn.append(attn_map.detach().cpu().numpy())

        masks = masks[:-1]
        mask_mid = masks[-1]
        attn_masks = attn_masks[:-1]
        attn_mask_mid = attn_masks[-1]

        x = self.mid_block1(x, mask_mid, t)
        x, attn_map = self.mid_attn(x) if self.att_type == "linear" else self.mid_attn(x, hids, hids,
                                                                                       attn_mask=attn_mask_mid)
        x = self.mid_block2(x, mask_mid, t)

        for i, (resnet1, resnet2, attn, upsample) in enumerate(self.ups):
            mask_up = masks.pop()
            attn_mask_up = attn_masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x, attn_map = attn(x) if self.att_type == "linear" else attn(x, hids, hids,
                                                                         attn_mask=attn_mask_up)
            x = upsample(x * mask_up)

        # Save p2f_map_score of last layer
        if show_attmap:
        #if t0[0] in attn_img_time and show_attmap and not self.training:    
            return_attn.append(attn_map.detach().cpu().numpy())

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)
        print(torch.mean(output))

        if return_attmap:
            """
            attn_map = torch.cat(
                [attn_map[:, i, :, :] for i in range(attn_map.size(1))], dim=-2)  # (b, d_q * l_q, d_m * h)
            """
            return (output * mask).squeeze(1), return_attn # ?? which attn_map ??
        else:
            return (output * mask).squeeze(1)

    def guide_sample(
            self,
            x: torch.Tensor,
            guid_mask: torch.Tensor,
            mask: torch.Tensor,
            mu: torch.Tensor,
            t,
            spk,
            melstyle,
            emo_label,
            align_len,
            align_mtx,
            guide_scale: float,
            tal=40,
            alpha=0.08,
            guide_pitch=True
    ):
        """
        Guide sample by single ref audio

        # set x to require_grad
        # set optimizer with x
        # set loss = attn_in_mask + attn_out_mask
        # get x_guided
        # get x_guided_score = unet(x_guided)

        Args:
            x ():
            guid_mask (l_k, 80):
            mask ():
            mu ():
            cond ():

        Returns:
            x_guided_score
        """
        loss = 0
        # set x to require_grad
        x_guided = x.clone().detach()
        x_guided.requires_grad = True

        # set optimizer with x
        optim = torch.optim.SGD([x_guided], lr=alpha)
        # set loss = attn_in_mask + attn_out_mask
        score_emo, attn_score = self.forward(x_guided,
                                       mask.detach(),
                                       mu.detach(),
                                       t.detach(),
                                       spk.detach(),
                                       psd=None,
                                       melstyle=melstyle.detach(),
                                       emo_label=emo_label.detach(),
                                       align_len=align_len,
                                       align_mtx=align_mtx.detach(),
                                       return_attmap=True
                                       ) # ...,  (b, l_q * d_q<-80, l_k) ?
        # align guidmask to attn_score
        if len(attn_score.shape) == 3:
            attn_score = attn_score[0]
        guid_mask = guid_mask.view(guid_mask.shape[0] * guid_mask.shape[1])
        # align guid_mask with length of key to length of query
        if len(guid_mask) < attn_score.shape[0]:
            p1d = (0, attn_score.shape[0] - len(guid_mask))
            guid_mask = F.pad(guid_mask, p1d, "constant", 0)
        else:
            guid_mask = guid_mask[:attn_score.shape[0]]

        # Sum of attn_score located inside/outside of pitchMask with guide_scale
        if guide_pitch:
            loss += -attn_score[guid_mask == 1].sum()
            loss += guide_scale * attn_score[guid_mask == 0].sum()
        else:
            loss += -attn_score[guid_mask == 0].sum()
            loss += guide_scale * attn_score[guid_mask == 1].sum()
        #loss.requires_grad = True
        loss.backward(retain_graph=False)
        print("loss.{}".format(loss))

        # get x_guided updated
        optim.step()
        # get x_guided_score = unet(x_guided)
        x_guided = x_guided.detach()
        return score_emo, x_guided

def align_cond_input_emomix(x, mu, spk, emoEmb):
    """
    emoEmb
    align cond according to emomix
    """
    pass
    #emoEmob = emoEmob.
    #x = torch.stack([x, mu, spk, emoEmob])


def align_cond_input(x, mu, spk, melstyle, emo_label, align_len, align_mtx, att_type):
    """
    Align x and conditions (spk, mel, emo_label) given linear nd cross attention
    - align_len: Padding conditions given aligning size
    - align_mtx: Converting conditions given aligning matrix
    """
    if align_len is not None:
        spk_align = align(spk, align_len, condtype="noseq")
        emo_label_align = align(emo_label, align_len, condtype="noseq")
        if align_mtx is not None:     # Only for inference
            melstyle = align(melstyle, align_len, align_mtx, condtype="seq")
    # Concatenate condition by different attention
    if att_type == "linear":
        x = torch.stack([x, mu, spk_align, emo_label_align, melstyle], 1)
        hids = None
    elif att_type == "cross":
        # To make the dimension of channel (3) * 80 same as hids (240) for classifier-free guidance
        # ??1. Expected samiliar result with different input makes training difficult.
        x = torch.stack([x, mu, mu], 1)
        #x = torch.stack([x, mu], 1)
        hids = torch.concat([spk_align, melstyle, emo_label_align], dim=1)
        # hids = torch.stack([spk_align, melstyle, emo_label_align], dim=1)
    else:
        print("bad att type!")
    return x, hids