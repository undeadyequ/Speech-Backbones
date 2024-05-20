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
from GradTTS.model.utils import align
import matplotlib.pyplot as plt
from bisect import bisect

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
        dims = [total_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        #print(in_out)
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        # Set unet
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                # MultiAttention3->frame2frame; MultiAttention2->bin2frame (simFrame2fame?)
                Residual(Rezero(LinearAttention(dim_out) if att_type == "linear" else
                                MultiAttention3(dim_out, self.enc_hid_dim, att_dim, heads, dim))),
                Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim) if att_type == "linear" else
                                        MultiAttention3(mid_dim, self.enc_hid_dim, att_dim, heads)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                Residual(Rezero(LinearAttention(dim_in) if att_type == "linear" else
                            MultiAttention3(dim_in, self.enc_hid_dim, att_dim, heads, dim / 2 if is_last else dim))),
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
                return_attmap=False
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
        # Regularize unet input1: xt and enc_hids
        x, hids = align_cond_input(x, mu, spk, melstyle, emo_label, align_len, align_mtx, self.att_type)

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
                cond_res, attn_amp = self.forward_to_unet(x, mask, hids, t, return_attmap=return_attmap)
            else:
                cond_res = self.forward_to_unet(x, mask, hids, t, return_attmap=return_attmap)
            b, c, d_q, l_q = x.shape
            hids = x.clone()  # <- NO condition
            hids = hids.view(b, -1, l_q)
            uncond_res = self.forward_to_unet(x, mask, hids, t)
            res = (1 + guidence_strength) * cond_res - guidence_strength * uncond_res
            if return_attmap:
                return res, attn_amp
        return res

    def forward_to_unet(self,
                        x,
                        mask,
                        hids,
                        t,
                        show_attmap=False,
                        return_attmap=False):
        # show att map when t in ATTN_MAP_SHOW_range
        ATTN_MAP_SHOW_range = (0.2, 0.24)  # <- check front (near 1) or back (near 0)
        ATTN_MAP_SHOW_FLAG = False
        if (torch.tensor(ATTN_MAP_SHOW_range[0], dtype=t.dtype, device=t.device)
                < t[0] < torch.tensor(ATTN_MAP_SHOW_range[1], dtype=t.dtype, device=t.device)) and show_attmap:
            t_show = t[0].detach().cpu().numpy()
            ATTN_MAP_SHOW_FLAG = True  # only show the attention of first unet layer.

        # Regularize Unet input2: Time projection
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.t_mlp(t)
        mask = mask.unsqueeze(1)
        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)   # (b, c, d, l) -> (b, c_out_i, d, l)
            x = resnet2(x, mask_down, t)   # -> (b, c_out_i, d, l)
            x, attn_map = attn(x) if self.att_type == "linear" else attn(x, key=hids, value=hids) #  -> (b, c_out_i, d, l)
            hiddens.append(x)
            x = downsample(x * mask_down)  # -> (b, c_out_i, d/2, l/2)
            masks.append(mask_down[:, :, :, ::2])

            #### show Attn image ####
            if ATTN_MAP_SHOW_FLAG:
                head_n = attn_map.shape[1]
                for h in range(head_n):
                    attn_np = attn_map.detach().cpu().numpy()
                    plt.imsave("temp/attn_head{}_{}.png".format(h, t_show), attn_np[0, h, :, :])
                ATTN_MAP_SHOW_FLAG = False
            #### END ####

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x, attn_map = self.mid_attn(x) if self.att_type == "linear" else self.mid_attn(x, hids, hids)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x, attn_map = attn(x) if self.att_type == "linear" else attn(x, hids, hids)
            x = upsample(x * mask_up)
        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        if return_attmap:
            attn_map = torch.cat(
                [attn_map[:, i, :, :] for i in range(attn_map.size(1))], dim=-2)  # (b, d_q * l_q, d_m * h)
            return (output * mask).squeeze(1), attn_map  # ?? which attn_map ??
        else:
            return (output * mask).squeeze(1)

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
            if i == 0 or mask_all_layer:  ## ?? good
                b, _, d_q, l_q = x.shape
                # generate 1st/2nd mask (e.g. mask left/right)
                #print("up x shape: {}".format(x.shape))
                temp_mask_left, temp_mask_right = create_left_right_mask(b, self.heads, d_q, l_q)
                x_left, attn_map_left = attn(x, key=hids1, value=hids1, attn_mask=temp_mask_left,
                              mask_value=temp_mask_value)  # -> (b, c_out_i, d, l)
                x_right, attn_map_right = attn(x, key=hids2, value=hids2, attn_mask=temp_mask_right,
                               mask_value=temp_mask_value)  # -> (b, c_out_i, d, l)
                x = x_left + x_right
            else:
                # only apply reference audio2 in other layers
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

            # compute temporal mask and get temporally interpolated x
            if i == 0 or mask_all_layer:
                b, _, d_q, l_q = x.shape
                #print("down x shape: {}".format(x.shape))
                temp_mask_left, temp_mask_right = create_left_right_mask(b, self.heads, d_q, l_q)
                x_left, attn_map_left = attn(x, key=hids1, value=hids1,
                                             attn_mask=temp_mask_right,
                                             mask_value=temp_mask_value)   # -> (b, c_out_i, d, l)
                x_right, attn_map_right = attn(x, key=hids2, value=hids2,
                                               attn_mask=temp_mask_left,
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
        Args:
            x:
            mask:
            mu:
            t:
            spk:
            melstyle1:
            emo_label1:
            pitch1:
            melstyle2:
            emo_label2:
            pitch2:
            align_len:
            align_mtx:
            guide_scale:
            tal:
            alpha:

        Returns:

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


def create_left_right_mask(b, heads, d_q, l_q, device="cuda"):
    if device == "cuda":
        temp_mask_left = torch.ones((b, heads, d_q * l_q, 1)).cuda()
        temp_mask_right = torch.ones((b, heads, d_q * l_q, 1)).cuda()
    else:
        temp_mask_left = torch.ones((b, heads, d_q * l_q, 1))
        temp_mask_right = torch.ones((b, heads, d_q * l_q, 1))
    temp_mask_left[:, :, :d_q * int(l_q * 0.5), :] = 0
    temp_mask_left = temp_mask_left > 0
    temp_mask_right[:, :, d_q * int(l_q * 0.5):, :] = 0
    temp_mask_right = temp_mask_right > 0
    return temp_mask_left, temp_mask_right


def create_utter_word_mask_freq(b, heads, d_q, l_q, emb_n,
                                ref_start, ref_end, target_start, target_end):
    """
    Create masks for utterance-level and word-level style transfering.
    """
    # create utter mask
    temp_mask_utter = torch.ones((b, heads, d_q * l_q, emb_n)).cuda()
    temp_mask_utter[:, :, target_start:target_end, :] = 0

    # create word mask
    temp_mask_word = torch.zeros((b, heads, d_q * l_q, emb_n)).cuda()
    temp_mask_word[:, :, target_start:target_end, ref_start:ref_end] = 1
    return temp_mask_utter, temp_mask_word


def create_low_high_mask_freq(b, heads, d_q, l_q,
                              freq_splitter="pitch"):
    """
    low high frequency mask
    """

    if freq_splitter == "pitch":
        split_mel_bin = change_freq_to_melbins(pitch_range[1])
    elif freq_splitter == "formant12":
        split_mel_bin = change_freq_to_melbins(f12_range[1])
    else:
        split_mel_bin = 0
        IOError("freq_splitter should be pitch or formant12")

    mask_low_freq = torch.zeros((b, heads, d_q * l_q, 1)).cuda()
    for i in l_q:
        mask_low_freq[:, :, i * d_q:i * ( d_q + split_mel_bin), :] = 1
    mask_high_freq = 1 - mask_low_freq
    return mask_low_freq, mask_high_freq

def align_cond_input(x, mu, spk, melstyle, emo_label, align_len, align_mtx, att_type):
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
        # In order to make the dim of channel (3) * 80 same as hids (240) for classifier-free guidance
        # ??1. Expected samiliar result with different input makes training difficult.
        x = torch.stack([x, mu, mu], 1)
        #x = torch.stack([x, mu], 1)

        hids = torch.concat([spk_align, melstyle, emo_label_align], dim=1)
        # hids = torch.stack([spk_align, melstyle, emo_label_align], dim=1)
    else:
        print("bad att type!")
    return x, hids



# https://iopscience.iop.org/article/10.1088/1742-6596/1153/1/012008/pdf
# freq_splitter = "pitch"
pitich_range_male = (100, 150)
pitch_range_female = (200, 250)
pitch_range = (90, 255)
non_pitch_range = (255, 5500)

# freq_splitter = "formant12"
f1_range = (650, 950)
f2_range = (1300, 1500)
f12_range = (650, 1500)

f3_range = (2500, 4000)
f4_range = (4500, 5500)  # not for sure
f34_range = (1500, 5500)

def change_freq_to_melbins(search_freq):
    freq_bins = librosa.mel_frequencies(n_mels=80)
    return bisect(freq_bins, search_freq)



# ------------------------- NOT USED-----------------------------
def create_left_right_mask_temp(b, heads, d_q, l_q):
    """
    left_mask_list = []
    right_mask_list = []
    for s in range(int(l_q * 0.5)):
        r_n = torch.arange(s, (l_q - 1) * d_q + s, d_q)
        left_mask_list.append(r_n)
    for s in range(l_q - int(l_q * 0.5)):
        r_n = torch.arange(l_q - s - 1, (l_q - 1) * d_q + l_q - s - 1, d_q)
        right_mask_list.append(r_n)

    left_mask_torch = torch.sort(torch.concat(left_mask_list, dim=1)).unsqueeze(-1)
    right_mask_list = torch.sort(torch.concat(right_mask_list, dim=1)).unsqueeze(-1)
    """

    temp_mask_left = torch.ones((b, heads, d_q * l_q, 1)).cuda()
    temp_mask_right = torch.ones((b, heads, d_q * l_q, 1)).cuda()
    temp_mask_left[:, :, :d_q * int(l_q * 0.5), :] = 0
    temp_mask_left = temp_mask_left > 0
    temp_mask_right[:, :, d_q * int(l_q * 0.5):, :] = 0
    temp_mask_right = temp_mask_right > 0
    return temp_mask_left, temp_mask_right

def create_pitch_bin_mask(wav_f,
                          mel_spectrogram,
                          n_mels=80,
                          fmin=0.0,
                          fmax=8000.0
                          ):
    """
    create pitch_bin mask given psd contours (Only in Inference)
    # change pitch_contour to pitch_bin_contour (denormalization to freq to bin)
    # pitch_bin_contour to pitch_bin_mask

    Args:
        wav_f:
        mel_spectrogram: (b, ?, 768)
        psd_contours: (b, 3, r)
        psd_mask : (b, 3)

    Returns:
        pitch_bin_mask = (b, c, d, l)

    """
    # Trim speech ?? no Need trim ?? -> if trim shorter than mel, no trim far greater than mel
    tg_dir = "/home/rosen/Project/FastSpeech2/preprocessed_data/ESD/TextGrid"
    wav_base = os.path.basename(wav_f).split(".")[0]
    tg_sub_dir = tg_dir + "/" + wav_base.split("_")[0]
    tg_f = tg_sub_dir + "/" + wav_base + ".TextGrid"

    pitch_contour = extract_pitch(wav_f, tg_f, need_trim=True)
    # TEMP: pad/remove pitch length to match with mel
    if len(mel_spectrogram.shape) == 3:
        mel_spectrogram = mel_spectrogram.squeeze()
    if len(pitch_contour) < mel_spectrogram.shape[0]:
        pitch_contour = np.append(pitch_contour, [0.0] * (len(mel_spectrogram) - len(pitch_contour)))
    elif len(pitch_contour) > mel_spectrogram.shape[0]:
        pitch_contour = pitch_contour[:mel_spectrogram.shape[0]]
    else:
        print("Pitch length match mel length")

    # change pitch_contour to pitch_bin_contour (denormalization -> ?)
    # pitch freq to mel bin given frequencies tuned to mel scale.
    freq_list = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax, htk=False)
    # Convert pitch hz to mel bin for each mel frame
    pitch_mel_bin_list = []  # (mel_n, )
    for pitch in pitch_contour:
        pitch_mel_bin = np.argmin(abs(freq_list - pitch))
        pitch_mel_bin_list.append(pitch_mel_bin)

    pitch_bin_mask = torch.zeros_like(mel_spectrogram)
    for fr_th in range(mel_spectrogram.shape[0]):
        # map the i-th frame to j-th pitch <- if len(mel) != len(pitch)
        #pit_th = pitmel_align_score[fr_th]
        fr_mel_bin = pitch_mel_bin_list[fr_th]
        if fr_mel_bin != 0:  # Only for pitch existed!
            pitch_bin_mask[fr_th, fr_mel_bin] = 1
    return pitch_bin_mask


def create_left_right_mask_score(b, heads, d_q, l_q, device="cuda"):
    if device == "cuda":
        temp_mask_left = torch.zeros((b, heads, d_q * l_q, 1)).cuda()
        temp_mask_right = torch.zeros((b, heads, d_q * l_q, 1)).cuda()
    else:
        temp_mask_left = torch.zeros((b, heads, d_q * l_q, 1))
        temp_mask_right = torch.zeros((b, heads, d_q * l_q, 1))

    temp_left_score = torch.linspace(0, 1, steps=d_q * int(l_q * 0.5)).unsqueeze(0).transpose(0, 1)
    temp_right_score = torch.linspace(0, 1, steps=d_q * l_q - d_q * int(l_q * 0.5)).unsqueeze(0).transpose(0, 1)

    temp_left_score = temp_left_score.repeat(b, heads, 1, 1)
    temp_right_score = temp_right_score.repeat(b, heads, 1, 1)

    temp_mask_left[:, :, :d_q * int(l_q * 0.5), :] = temp_left_score
    temp_mask_right[:, :, d_q * int(l_q * 0.5):, :] = temp_right_score
    return temp_mask_left, temp_mask_right