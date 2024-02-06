
from einops import rearrange
import math
import torch
import torch.nn.functional as F
from GradTTS.model.base import BaseModule
from GradTTS.model.diffusion import *
from GradTTS.model.utils import align
import matplotlib.pyplot as plt


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
                 heads=4
                 ):
        """

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
        # sample channel
        total_dim = sample_channel_n + (1 if n_spks > 1 else 0)  # 4 include spk, emo, psd, x
        # enc_hid channel


        dims = [total_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                # Sample
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                # att_dim=int(80 / 2 ** ind) ???
                Residual(Rezero(LinearAttention(dim_out) if att_type == "linear" else
                                MultiAttention2(dim_out, self.enc_hid_dim, att_dim, heads))),  # Rezero is Not usefull ??  or MultiAttention
                Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        # enc_hid (ResnetBlock)...
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim) if att_type == "linear" else
                                        MultiAttention2(mid_dim, self.enc_hid_dim, att_dim, heads)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        # enc_hid (Downsample)...

        ft_dims = int(80 / 2 ** (len(dim_mults) - 1))
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
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
                align_mtx=None
                ):
        """

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
        # Regulize Unet input1: xt and enc_hids
        if align_len is not None:
            spk_align = align(spk, align_len, condtype="noseq")
            emo_label_align = align(emo_label, align_len, condtype="noseq")
            if align_mtx is not None:   # Only for inference
                melstyle = align(melstyle, align_len, align_mtx, condtype="seq")

        # Concatenate condition by different attention
        if self.att_type == "linear":
            x = torch.stack([x, mu, spk_align, emo_label_align], 1)
            hids = None
        elif self.att_type == "cross":
            x = torch.stack([x, mu], 1)
            hids = torch.concat([spk_align, melstyle, emo_label_align], dim=1)
            #hids = torch.stack([spk_align, melstyle, emo_label_align], dim=1)
        else:
            print("bad att type!")

        # Judge whether show att map
        ATTN_MAP_SHOW_range = (0.2, 0.24)  # <- check front (near 1) or back (near 0)
        ATTN_MAP_SHOW_FLAG = False
        if (torch.tensor(ATTN_MAP_SHOW_range[0], dtype=t.dtype, device=t.device)
                < t[0] < torch.tensor(ATTN_MAP_SHOW_range[1], dtype=t.dtype, device=t.device)):
            T_SHOW = t[0].detach().cpu().numpy()
            ATTN_MAP_SHOW_FLAG = True

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
            # show image
            if ATTN_MAP_SHOW_FLAG:
                head_n = attn_map.shape[1]
                for h in range(head_n):
                    attn_np = attn_map.detach().cpu().numpy()
                    plt.imsave("temp/attn_head{}_{}.png".format(h, T_SHOW), attn_np[0, h, :, :])
                ATTN_MAP_SHOW_FLAG = False

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
        return (output * mask).squeeze(1)

    def forward_temporal_interp(self,
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
                align_mtx=None
                ):
        """

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
        # Regulize Unet input1: xt and enc_hids
        _, hids1 = align_cond_input(x, mu, spk, melstyle1, emo_label1, align_len, align_mtx, self.att_type)
        x, hids2 = align_cond_input(x, mu, spk, melstyle2, emo_label2, align_len, align_mtx, self.att_type)

        # Regulize Unet input2: Time projection
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.t_mlp(t)

        mask = mask.unsqueeze(1)
        hiddens = []
        masks = [mask]

        temp_mask_value = 0
        MASK_ALL_UNET_LAYER = False

        for i, (resnet1, resnet2, attn, downsample) in enumerate(self.downs):         # c d l -> c*2 d/2 l/2 as layer increase
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)   # (b, c, d, l) -> (b, c_out_i, d, l)
            x = resnet2(x, mask_down, t)   # -> (b, c_out_i, d, l)

            # compute temporal mask and get temporally interpolated x
            if i == 0 or MASK_ALL_UNET_LAYER:
                b, _, d_q, l_q = x.shape
                temp_mask_left, temp_mask_right = create_left_right_mask(b, self.heads, d_q, l_q)
                x_left, attn_map_left = attn(x, key=hids1, value=hids1, attn_mask=temp_mask_left,
                              mask_value=temp_mask_value)  # -> (b, c_out_i, d, l)
                x_right, attn_map_right = attn(x, key=hids2, value=hids2, attn_mask=temp_mask_right,
                               mask_value=temp_mask_value)  # -> (b, c_out_i, d, l)
                x = x_left + x_right
            else:
                temp_mask_left, temp_mask_right = None, None
                x, attn_map = attn(x, key=hids2, value=hids2, attn_mask=temp_mask_right,
                               mask_value=temp_mask_value)  # -> (b, c_out_i, d, l)

            hiddens.append(x)
            x = downsample(x * mask_down)  # -> (b, c_out_i, d/2, l/2)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        x = self.mid_block1(x, mask_mid, t)
        if MASK_ALL_UNET_LAYER:
            x_left, attn_map_left = self.mid_attn(x, hids1, hids1, attn_mask=temp_mask_left, mask_value=temp_mask_value)
            x_right, attn_map_right = self.mid_attn(x, hids2, hids2, attn_mask=temp_mask_right, mask_value=temp_mask_value)
            x = x_left + x_right
        else:
            x = self.mid_attn(x, hids2, hids2, attn_mask=temp_mask_right, mask_value=temp_mask_value)
        x = self.mid_block2(x, mask_mid, t)


        for i, (resnet1, resnet2, attn, upsample) in enumerate(self.ups):
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)

            # compute temporal mask and get temporally interpolated x
            if i == 0 or MASK_ALL_UNET_LAYER:
                b, _, d_q, l_q = x.shape
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


def create_left_right_mask_temp(b, heads, d_q, l_q):
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

    temp_mask_left = torch.ones((b, heads, d_q * l_q, 1)).cuda()
    temp_mask_right = torch.ones((b, heads, d_q * l_q, 1)).cuda()
    temp_mask_left[:, :, :d_q * int(l_q * 0.5), :] = 0
    temp_mask_left = temp_mask_left > 0
    temp_mask_right[:, :, d_q * int(l_q * 0.5):, :] = 0
    temp_mask_right = temp_mask_right > 0
    return temp_mask_left, temp_mask_right


def align_cond_input(x, mu, spk, melstyle, emo_label, align_len, align_mtx, att_type):
    if align_len is not None:
        spk_align = align(spk, align_len, condtype="noseq")
        emo_label_align = align(emo_label, align_len, condtype="noseq")
        if align_mtx is not None:     # Only for inference
            melstyle = align(melstyle, align_len, align_mtx, condtype="seq")
    # Concatenate condition by different attention
    if att_type == "linear":
        x = torch.stack([x, mu, spk_align, emo_label_align], 1)
        hids = None
    elif att_type == "cross":
        x = torch.stack([x, mu], 1)
        hids = torch.concat([spk_align, melstyle, emo_label_align], dim=1)
        # hids = torch.stack([spk_align, melstyle, emo_label_align], dim=1)
    else:
        print("bad att type!")
    return x, hids