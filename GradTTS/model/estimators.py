
from einops import rearrange
import math
import torch
import torch.nn.functional as F
from GradTTS.model.base import BaseModule
from GradTTS.model.diffusion import *

class GradLogPEstimator2dCond(BaseModule):
    def __init__(self,
                 dim,
                 dim_mults=(1, 2, 4),
                 groups=8,
                 n_spks=None,
                 spk_emb_dim=64,
                 n_feats=80,
                 pe_scale=1000,
                 emo_emb_dim=768,
                 att_type="linear"
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

        if n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4), Mish(),
                                               torch.nn.Linear(spk_emb_dim * 4, n_feats))
        self.emo_mlp = torch.nn.Sequential(torch.nn.Linear(emo_emb_dim, emo_emb_dim), Mish(),
                                           torch.nn.Linear(emo_emb_dim, n_feats))
        self.psd_mlp = torch.nn.Sequential(torch.nn.Linear(3, 16), Mish(),
                                           torch.nn.Linear(16, n_feats))

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(),
                                       torch.nn.Linear(dim * 4, dim))

        #
        if self.att_type == "linear":
            total_dim = 4 + (1 if n_spks > 1 else 0)  # 4 include spk, emo, psd, x
        elif self.att_type == "crossAtt":
            total_dim = 2 + (1 if n_spks > 1 else 0)  # 4 include spk, emo, psd, x
        else:
            total_dim = 0
            print("input linear or crossAtt")
        dims = [total_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                Residual(Rezero(LinearAttention(dim_out) if att_type == "linear" else
                                torch.nn.MultiheadAttention(embed_dim=128, num_heads=8))),
                Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim) if att_type == "linear" else
                                        torch.nn.MultiheadAttention(embed_dim=128, num_heads=8)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                Residual(Rezero(LinearAttention(dim_in) if att_type == "linear" else
                                torch.nn.MultiheadAttention(embed_dim=128, num_heads=8))),
                Upsample(dim_in)]))
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask, mu, t, spk=None, txt=None, emo_label=None, psd=None):
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
        Returns:
            output: (b, 80, mel_len)
        """
        # Project conditioning value to 80 dims for concatenating with X
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)
        if not isinstance(emo_label, type(None)):
            emo_label = self.emo_mlp(emo_label)  # (b, 1, 80)
        if not isinstance(psd, type(None)):
            psd = [p1.to(torch.float) for p1 in psd]
            psd = torch.stack(psd, 0)
            psd = torch.permute(psd, (1, 0, 2))
            if self.att_type == "linear":
                psd = torch.permute(psd, (0, 2, 1))
                psd = self.psd_mlp(psd)  # (b, word_len, 80)
                psd = torch.permute(psd, (0, 2, 1))  # (b, 80, word_len)

        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)  # (b, 1, 4*dim)

        hids = []

        # create sequential dimension for speaker and emotion
        if self.n_spks >= 2 and spk is not None:
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.stack([x, mu, s], 1)  # (b, 3, 80, mel_len)
        else:
            x = torch.stack([mu, x], 1)     # (b, 2, 80, mel_len)
        if emo_label is not None:
            if self.att_type == "linear":   # ? Condition method: directly
                emo_label = emo_label.unsqueeze(-1).repeat(1, 1, x.shape[-1]).unsqueeze(1)  # (b, 1, 80, mel_len)
                x = torch.cat([x, emo_label], 1)  # # (b, 3 or4, 80, mel_len)
            else:
                hids = emo_label

        # align sequential dimension for psd
        if psd is not None:
            if self.att_type == "linear":
                # [temporary] pad psd from word_len to mel_len(100)
                if True:
                    mel_len = x.shape[-1]
                    word_len = psd.shape[-1]
                    left_pad = int((mel_len - word_len)/2)
                    right_pad = mel_len - left_pad - word_len
                    p1d = (left_pad, right_pad)
                    psd = F.pad(psd, p1d, "constant", 0)  # (b, 80, word_len) -> # (b, 80, mel_len)
                    psd = psd.unsqueeze(1)   # (b, 1, 80, mel_len)
                x = torch.cat([x, psd], 1)   # (b, 4/5, 80, mel_len)
                # [permanent?] align word and mel length in Dataset class ...
            else:
                if len(hids) != 0:
                    hids = hids.unsqueeze(1).repeat(1, psd.shape[1], 1)
                    hids = torch.cat([hids, psd], 1)
                else:
                    hids = psd

        mask = mask.unsqueeze(1)
        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            if self.att_type == "linear":
                x = attn(x)
            else:
                x = attn(query=x, key=hids, value=hids)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        x = self.mid_block1(x, mask_mid, t)
        if self.att_type == "linear":
            x = self.mid_attn(x)
        else:
            x = self.mid_attn(query=x, key=emo_label, value=emo_label)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            if self.att_type == "linear":
                x = attn(x)
            else:
                x = attn(query=x, key=emo_label, value=emo_label)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)
