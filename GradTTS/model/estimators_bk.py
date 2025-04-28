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
                attn_mask=None
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
            mask:(b, mel_len)
            mu:  (b, 80, mel_len)
            t:   (b, 64)
            spk: (b, spk_emb)
            emo_label: (b, 1, emo_num)
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
            res = self.forward_to_unet(x, mask, hids, t)
        # Conditional sampling with weighted noise of uncond/cond
        else:
            if return_attmap:
                cond_res, attn_amp = self.forward_to_unet(x, mask, hids, t,
                                                          return_attmap=return_attmap,
                                                          attn_mask=attn_mask)
            else:
                cond_res = self.forward_to_unet(x, mask, hids, t, return_attmap=return_attmap,
                                                attn_mask=attn_mask)
            if self.p_uncond != 0.0:
                b, c, d_q, l_q = x.shape
                hids = x.clone()  # <- NO condition
                hids = hids.view(b, -1, l_q)
                uncond_res = self.forward_to_unet(x, mask, hids, t, attn_mask=attn_mask)
                res = (1 + guidence_strength) * cond_res - guidence_strength * uncond_res
            else:
                res = cond_res

            if return_attmap:
                return res, attn_amp
        return res

    def forward_to_unet(self,
                        x,
                        mask,
                        hids,
                        t,
                        show_attmap=True,
                        return_attmap=False,
                        attn_mask=None
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
        # show att map when t in ATTN_MAP_SHOW_range
        #ATTN_MAP_SHOW_range = (0.2, 0.24)  # <- check front (near 1) or back (near 0)
        input = torch.clone(x)

        attn_img_time_ind = [0, 10, 20, 30, 40, 49]  # <- check front (near 1) or back (near 0)
        n_timesteps = 50
        attn_img_time = torch.tensor(
            [(1.0 - (ind + 0.5) * (1.0 / n_timesteps)) for ind in attn_img_time_ind], dtype=t.dtype, device=t.device
        )
        ATTN_MAP_SHOW_FLAG = False
        #if (torch.tensor(ATTN_MAP_SHOW_range[0], dtype=t.dtype, device=t.device)
        #        < t[0] < torch.tensor(ATTN_MAP_SHOW_range[1], dtype=t.dtype, device=t.device)) and show_attmap:
        if t[0] in attn_img_time and show_attmap:
            t_show = t[0].detach().cpu().numpy()
            ATTN_MAP_SHOW_FLAG = True  # only show the attention of first unet layer.

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

            #### show Attn image ####
            if ATTN_MAP_SHOW_FLAG:
                # Show each p2f attention of each phoneme on each head
                head_n = attn_map.shape[1]
                for h in range(head_n):
                    if attn_map.shape[2] % 80 != 0:
                        raise IOError("attn_map shape should be divide by 80: {}".format(attn_map.shape))
                    phoneme_num = int(attn_map.shape[2] / 80)
                    for phoneme_index in range(phoneme_num):
                        attn_np = attn_map.detach().cpu().numpy()
                        p2f_attn = attn_np[0, h, phoneme_index * 80: (phoneme_index + 1) * 80, :]
                        attn_time_dir = "temp/attn_time{}".format(t_show)
                        if not Path(attn_time_dir).is_dir():
                            Path(attn_time_dir).mkdir(parents=True, exist_ok=True)
                        plt.imsave(attn_time_dir + "/head{}_phone{}.png".format(h, phoneme_index),
                                   p2f_attn)
                ATTN_MAP_SHOW_FLAG = False
            #### END ####

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

        # check mid value
        CHECK_ATTN_OUT = False
        if CHECK_ATTN_OUT:
            attn_map_show = attn_map.transpose(2, 3).detach().cpu().numpy()
            output_show = x.detach().cpu().numpy()
            attnin_show = input.detach().cpu().numpy()
            save_plot(attn_map_show[0, 0, :, 8::80],
                      "/home/rosen/Project/Speech-Backbones/GradTTS/{}_channel{}.png".format("attn_map", 0),
                      size=(12, 12))
            save_plot(output_show[0, 0, :, :],
                      "/home/rosen/Project/Speech-Backbones/GradTTS/{}_channel{}.png".format("attn_out", 0))
            save_plot(attnin_show[0, 0, :, :],
                      "/home/rosen/Project/Speech-Backbones/GradTTS/{}_channel{}.png".format("attn_in", 0))

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        if return_attmap:
            attn_map = torch.cat(
                [attn_map[:, i, :, :] for i in range(attn_map.size(1))], dim=-2)  # (b, d_q * l_q, d_m * h)
            return (output * mask).squeeze(1), attn_map  # ?? which attn_map ??
        else:
            return (output * mask).squeeze(1)

    def forward_to_unet_mix(
            self,
            x,
            mask,
            hids1,
            hids2,
            t,
            attn_mask1,
            attn_mask2
    ):
        """
        Forward to unet model with single hids <- modify to unet_2d_cond_speech class in v2
        mask:  mask for length (l) indicator of x (b, c, d, l)
        attn_mask1: mask for attn (b, c, d * l, l2)
        """
        input_x = torch.clone(x)
        hiddens = []
        mask = mask.unsqueeze(1)  # x mask
        masks = [mask]
        attn_masks1 = [attn_mask1]  # attn mask
        attn_masks2 = [attn_mask2]

        # Down
        for i, (resnet1, resnet2, attn, downsample) in enumerate(self.downs):
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)  # (b, c, d, l) -> (b, c_out_i, d, l)
            x = resnet2(x, mask_down, t)  # -> (b, c_out_i, d, l)
            torch.set_printoptions(threshold=10_000)
            x1, attn_map1 = attn(x, hids1, hids1, attn_mask=attn_mask1, mask_value=0)
            x2, attn_map2 = attn(x, hids2, hids2, attn_mask=attn_mask2, mask_value=0)
            x = x1 + x2
            #x = x1

            # -> (b, c_out_i, d, l), (b, h, l_q * d_q, l_k)
            hiddens.append(x)

            x = downsample(x * mask_down)  # -> (b, c_out_i, d/2, l/2)
            masks.append(mask_down[:, :, :, ::2])    #

            attn_mask1 = attn_mask1[:, :, ::4, :]  # Q: d/2 * l/2 = d*l*1/4   KV: No changed
            attn_mask2 = attn_mask2[:, :, ::4, :]  # Q: d/2 * l/2 = d*l*1/4   KV: No changed
            attn_masks1.append(attn_mask1)
            attn_masks2.append(attn_mask2)

            #### show Attn image ####
            """
            if ATTN_MAP_SHOW_FLAG:
                show_attn_img(attn_map1, t[0].detach().cpu().numpy())
                ATTN_MAP_SHOW_FLAG = False
            """

        masks = masks[:-1]
        mask_mid = masks[-1]
        attn_masks1 = attn_masks1[:-1]
        attn_mask_mid1 = attn_masks1[-1]

        attn_masks2 = attn_masks2[:-1]
        attn_mask_mid2 = attn_masks2[-1]

        # Mid
        x = self.mid_block1(x, mask_mid, t)
        x1, attn_map1 = self.mid_attn(x, hids1, hids1, attn_mask=attn_mask_mid1)
        x2, attn_map2 = self.mid_attn(x, hids2, hids2, attn_mask=attn_mask_mid2)
        x = x1 + x2
        #x = x1
        x = self.mid_block2(x, mask_mid, t)

        # Up
        for i, (resnet1, resnet2, attn, upsample) in enumerate(self.ups):
            mask_up = masks.pop()
            attn_mask_up1 = attn_masks1.pop()
            attn_mask_up2 = attn_masks2.pop()

            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x1, attn_map1 = attn(x, hids1, hids1, attn_mask=attn_mask_up1, mask_value=0)
            x2, attn_map2 = attn(x, hids2, hids2, attn_mask=attn_mask_up2, mask_value=0)
            x = x1 + x2
            #x = x1
            x = upsample(x * mask_up)

        # Check mid value
        CHECK_ATTN_OUT = False
        if CHECK_ATTN_OUT:
            pass
            #show_attn_out(attn_map1, x, input_x)
            # show_attn_out(attn_map2, x, input_x)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)

    def reverse_mix(self,
                    x,
                    mask,
                    mu,
                    t,
                    spk1=None,
                    melstyle1=None,
                    emo_label1=None,
                    attn_mask1=None,
                    spk2=None,
                    melstyle2=None,
                    emo_label2=None,
                    attn_mask2=None,
                    align_len=None,
                    align_mtx=None,
                    guidence_strength=3.0,
                    return_attmap=False,
                    ):
        """
        reverse with two reference speech (same speaker) (only inference)
        align_len: pre-defined reference len to be aligned (reference refer to spk, mel, emo_label))
        align_mtx: reference len that computed by align_mtxt
        """
        # align and combine reference style
        _, hids1 = align_cond_input(x, mu, spk1, melstyle1, emo_label1, align_len, align_mtx, self.att_type)
        x, hids2 = align_cond_input(x, mu, spk2, melstyle2, emo_label2, align_len, align_mtx, self.att_type)

        # convert attn_mask after align
        if attn_mask1 is not None:
            attn_mask1 = align_a2b_padcut(attn_mask1, align_len, False)
        if attn_mask2 is not None:
            attn_mask2 = align_a2b_padcut(attn_mask2, align_len, True)

        # Regulize Unet input2: Time projection
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.t_mlp(t)

        output = self.forward_to_unet_mix(
            x,
            mask,
            hids1,
            hids2,
            t,
            attn_mask1,
            attn_mask2
        )
        return output


    def forward_temp_interp(self,
                            x,
                            mask,
                            mu,
                            t,
                            spk=None,
                            melstyle1=None,
                            emo_label1=None,
                            melstyle2=None,
                            emo_label2=None,
                            align_len=None,
                            align_mtx=None,
                            mask_all_layer=True,
                            temp_mask_value=0
                            ):
        """
        interpolating inference
        att_type option
            linear (self-attention):
                x = x + emo + psd
            CrossAtt (Not used):
                q = x
                k, v = emo + psd ? (No text information)

        Args:
            x:   (b, 80, mel_len)   # mel_len = 100 (fixed)
            mask:(b, mel_len)
            mu:  (b, 80, mel_len)
            t:   (b, 64)
            spk: (b, spk_emb)
            emo_label: (b, 1, emo_num)
            psd [tuple]: ((b, word_len), (b, word_len), (b, word_len))
            melstyle: (b, 80, mel_len)
            align_mtx

        Returns:
            output: (b, 80, mel_len)
        """
        #print("start estimator prediction!")
        # Regulize Unet input1: xt and enc_hids
        _, hids1 = align_cond_input(x, mu, spk, melstyle1, emo_label1, align_len, align_mtx, self.att_type)
        x, hids2 = align_cond_input(x, mu, spk, melstyle2, emo_label2, align_len, align_mtx, self.att_type)

        # Regulize Unet input2: Time projection
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.t_mlp(t)

        mask = mask.unsqueeze(1)
        hiddens = []
        masks = [mask]

        
        for i, (resnet1, resnet2, attn, downsample) in enumerate(self.downs):         # c d l -> c*2 d/2 l/2 as layer increase
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)   # (b, c, d, l) -> (b, c_out_i, d, l)
            x = resnet2(x, mask_down, t)   # -> (b, c_out_i, d, l)

            # compute temporal mask and get temporally interpolated x
            if i == 0 or mask_all_layer:
                b, _, d_q, l_q = x.shape
                # generate 1st/2nd mask (e.g. mask left/right)
                #print("up x shape: {}".format(x.shape))
                temp_mask_left, temp_mask_right = create_left_right_mask(b, self.heads, d_q, l_q)   # [b, h, l_q, 1]
                x_left, attn_map_left = attn(x, key=hids1, value=hids1, attn_mask=temp_mask_left,
                                             mask_value=temp_mask_value)  # -> (b, c_out_i, d, l)
                x_right, attn_map_right = attn(x, key=hids2, value=hids2, attn_mask=temp_mask_right,
                                               mask_value=temp_mask_value)  # -> (b, c_out_i, d, l)
                x = x_left + x_right
            else:
                # only apply reference audio2 in bk layers
                temp_mask_left, temp_mask_right = None, None
                x, attn_map = attn(x, key=hids2, value=hids2)  # -> (b, c_out_i, d, l)
            hiddens.append(x)
            x = downsample(x * mask_down)  # -> (b, c_out_i, d/2, l/2)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        x = self.mid_block1(x, mask_mid, t)
        if mask_all_layer:
            x_left, attn_map_left = self.mid_attn(x, hids1, hids1, attn_mask=temp_mask_left, mask_value=temp_mask_value)
            x_right, attn_map_right = self.mid_attn(x, hids2, hids2, attn_mask=temp_mask_right, mask_value=temp_mask_value)
            x = x_left + x_right
        else:
            x, _ = self.mid_attn(x, hids2, hids2)
        x = self.mid_block2(x, mask_mid, t)

        for i, (resnet1, resnet2, attn, upsample) in enumerate(self.ups):
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)

            # Compute temporal mask and get temporally interpolated x
            if i == 0 or mask_all_layer:
                b, _, d_q, l_q = x.shape
                #print("down x shape: {}".format(x.shape))
                temp_mask_left, temp_mask_right = create_left_right_mask(b, self.heads, d_q, l_q)
                x_left, attn_map_left = attn(x, key=hids1, value=hids1,
                                             attn_mask=temp_mask_left,
                                             mask_value=temp_mask_value)   # -> (b, c_out_i, d, l)
                x_right, attn_map_right = attn(x, key=hids2, value=hids2,
                                               attn_mask=temp_mask_right,
                                               mask_value=temp_mask_value)  # -> (b, c_out_i, d, l)
                x = x_left + x_right
            else:
                temp_mask_left, temp_mask_right = None, None
                x, attn_map = attn(x, key=hids2, value=hids2, attn_mask=temp_mask_left,
                               mask_value=temp_mask_value)  # -> (b, c_out_i, d, l)
            x = upsample(x * mask_up)
        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)
        return (output * mask).squeeze(1)

    def forward_freq_interp(self,
                            x,
                            mask,
                            mu,
                            t,
                            spk=None,
                            melstyle1=None,
                            emo_label1=None,
                            pitch1=None,
                            melstyle2=None,
                            emo_label2=None,
                            pitch2=None,
                            align_len=None,
                            align_mtx=None,
                            guide_scale=1.0,
                            tal_right=0.8,
                            tal_left=0.6,
                            alpha=0.08,
                            ):
        """
        Reference
        """
        if t > tal_right:
            print("Refer to reference1!")
            # get guid mask from melstyle or prosodic contours ?
            guid_mask = create_pitch_bin_mask(
                pitch1,
                melstyle1
            )  # (l_k1, 80)
            # train tunable noise
            score, x_guided = self.guide_sample(
                x,
                guid_mask,
                mask,
                mu,
                t,
                spk,
                melstyle1,
                emo_label1,
                align_len,
                align_mtx,
                guide_scale,
                alpha,
                guide_pitch=True
            )
            print("x and x_guided is equal?: {}".format(torch.equal(x, x_guided)))
        elif t <= tal_right and t > tal_left:
            print("Refer to reference2!")
            guid_mask = create_pitch_bin_mask(
                pitch1,
                melstyle1
            )  # (l_k1, 80)
            # train tunable noise
            score, x_guided = self.guide_sample(
                x,
                guid_mask,
                mask,
                mu,
                t,
                spk,
                melstyle2,
                emo_label2,
                align_len,
                align_mtx,
                guide_scale,
                alpha,
                guide_pitch=False
            )
            print("x and x_guided is equal?: {}".format(torch.equal(x, x_guided)))
        else:
            with torch.no_grad():
                score = self.forward(
                    x,
                    mask,
                    mu,
                    t,
                    spk,
                    psd=None,
                    melstyle=melstyle2,
                    emo_label=emo_label2,
                    align_len=align_len,
                    align_mtx=align_mtx) # ?? Any problem ??
                x_guided = x
        return score, x_guided

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