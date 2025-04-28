# References:
# https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/transformer.py
# https://github.com/jaywalnut310/vits/blob/main/attentions.py
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.constants import troy_ounce

from GradTTS.model.estimator.stditBlockAttention import MultiHeadAttention, MultiHeadAttentionCross
from GradTTS.model.mocha import MoChA
from GradTTS.eval_util import get_quantile
from GradTTS.model.util_mask_process import generate_diagal_fromMask
from GradTTS.model.utils import search_ma

torch.set_printoptions(threshold=100000)
selected_channel = list(torch.randint(0, 256, (10,)))


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)
        self.act1 = nn.SiLU(inplace=True)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask



class DiTConVCrossBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_channels, filter_channels, num_heads, kernel_size=3, p_dropout=0.1, gin_channels=0):
        super().__init__()
        # self
        self.norm1 = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.attn = MultiHeadAttention(hidden_channels, hidden_channels, num_heads, p_dropout)
        self.norm2 = nn.LayerNorm(hidden_channels, elementwise_affine=False)

        # cross
        self.norm3 = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.attn2 = MultiHeadAttentionCross(hidden_channels, hidden_channels, num_heads, p_dropout)
        self.rnorm = nn.LayerNorm(hidden_channels, elementwise_affine=False)

        # FFN
        self.mlp = FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(gin_channels, hidden_channels) if gin_channels != hidden_channels else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_channels, 7 * hidden_channels, bias=True)
        )

    def forward(self, x, c, r=None, x_mask=None, muy_dur=None, ref_dur=None, attnCross=None):
        """
        Args:
            x : [batch_size, channel, time]
            c : [batch_size, channel]
            r: [batch_size, channel, time]
            x_mask : [batch_size, 1, time]
        return the same shape as x
        """

        # self
        x = x * x_mask
        attn_mask_self = x_mask.unsqueeze(1) * x_mask.unsqueeze(-1)  # shape: [batch_size, 1, time, time]
        attn_mask_self = torch.zeros_like(attn_mask_self).masked_fill(attn_mask_self == 0, -torch.finfo(x.dtype).max)

        # adaLN
        shift_msa, scale_msa, gate_msa, gate_mocha, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).unsqueeze(2).chunk(7, dim=1)  # shape: [batch_size, channel, 1]
        x_norm = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x_self = self.attn(self.modulate(x_norm, shift_msa, scale_msa),
                                     attn_mask_self) * x_mask
        x = x + gate_msa * x_self

        # CrossAttn
        if attnCross is None:
            attn_mask_cross = x_mask.repeat(1, r.shape[2], 1).transpose(1, 2)
        else:
            attn_mask_cross = attnCross
        ## converting mask
        attn_mask_cross = torch.zeros_like(attn_mask_cross).masked_fill(attn_mask_cross == 0,
                                                                          -torch.finfo(x.dtype).max)  # 0(needtomask) -> infi, 1 -> 0

        r = self.rnorm(r.transpose(1, 2)).transpose(1, 2)  # (b, hid_dim, cut_l)
        x_cross, attn_map = self.attn2(x=x, c=r, attn_mask=attn_mask_cross)   # attn_map: (b)

        x = x + gate_mocha * x_cross * x_mask  # b, hid_dim, cut_l

        # FFN
        x = x + gate_mlp * self.mlp(self.modulate(self.norm2(x.transpose(1, 2)).transpose(1, 2), shift_mlp, scale_mlp),
                                    x_mask)
        return x, attn_map

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift

class DiTConVMochaBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_channels, filter_channels, num_heads, kernel_size=3, p_dropout=0.1, gin_channels=0,
                 monotonic_approach="no", **mochaAttn_kwargs):
        super().__init__()
        self.monotonic_approach = monotonic_approach

        # self
        self.norm1 = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.attn = MultiHeadAttention(hidden_channels, hidden_channels, num_heads, p_dropout)
        self.norm2 = nn.LayerNorm(hidden_channels, elementwise_affine=False)

        # mocha
        self.norm3 = nn.LayerNorm(hidden_channels, elementwise_affine=False)

        if self.monotonic_approach == "mocha":
            self.attn2 = MoChA(**mochaAttn_kwargs)
        else:
            self.attn2 = MultiHeadAttentionCross(hidden_channels, hidden_channels, num_heads, p_dropout)
        self.rnorm = nn.LayerNorm(hidden_channels, elementwise_affine=False)

        # FFN
        self.mlp = FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(gin_channels, hidden_channels) if gin_channels != hidden_channels else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_channels, 7 * hidden_channels, bias=True)
        )

    def forward(self, x, c, r=None, x_mask=None, muy_dur=None, ref_dur=None, attnCross=None):
        """
        Args:
            x : [batch_size, channel, time]
            c : [batch_size, channel]
            r: [batch_size, channel, time]
            x_mask : [batch_size, 1, time]
        return the same shape as x
        """

        # self
        x = x * x_mask
        attn_mask_self = x_mask.unsqueeze(1) * x_mask.unsqueeze(-1)  # shape: [batch_size, 1, time, time]
        attn_mask_self = torch.zeros_like(attn_mask_self).masked_fill(attn_mask_self == 0, -torch.finfo(x.dtype).max)

        ttt = self.adaLN_modulation(c).unsqueeze(2).chunk(7, dim=1)
        shift_msa, scale_msa, gate_msa, gate_mocha, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).unsqueeze(2).chunk(7, dim=1)  # shape: [batch_size, channel, 1]
        x_norm = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x_self = self.attn(self.modulate(x_norm, shift_msa, scale_msa),
                                     attn_mask_self) * x_mask
        x = x + gate_msa * x_self

        #print("gate_msa mean:", torch.mean(torch.abs(gate_msa)))
        #print("gate_mocha mean:", torch.mean(torch.abs(gate_mocha)))
        #print("gate_mlp mean:", torch.mean(torch.abs(gate_mlp)))

        # quantile
        #print("x_input:", get_quantile(x[:, selected_channel, :], dim=2))
        #print("x_self:", get_quantile(x_self[:, selected_channel, :], dim=2))

        # mean
        #print("x_norm:", torch.mean(x_norm[:, selected_channel, :], dim=2))
        #print("x_input:", torch.mean(x[:, selected_channel, :], dim=2))
        #print("x_self:", torch.mean(x_self[:, selected_channel, :], dim=2))

        #print("scale_msa: ", scale_msa[:, selected_channel, :])
        #print("gate_msa: ", gate_msa[:, selected_channel, :])
        #print("shift_msa: ", shift_msa[:, selected_channel, :])

        # mocha
        if self.monotonic_approach == "mocha":
            mode = "parallel" if self.training else "hard"
            # r: (b, hid_dim, cut_l), x: (b, hid_dim, cut_l), x_mask: (b, 1, cut_l)
            x_mask_dupli = x_mask.repeat(1, r.shape[2], 1).transpose(1, 2)  # (qlen, klen)   -> do not mask reference speech
            r = self.rnorm(r.transpose(1, 2)).transpose(1, 2)  # (b, hid_dim, cut_l)
            #print("r:", get_quantile(r.transpose(1, 2)[:, selected_channel, :], dim=2))
            #print("r:", torch.mean(r[:, selected_channel, :], dim=2))
            mocha_res = self.attn2(key=r.transpose(1, 2), value=r.transpose(1, 2),
                                   query=self.norm3(x.transpose(1, 2)), mask=x_mask_dupli, mode=mode)
            x_cross, alfpha, stats = mocha_res  # x_out: (b, cut_l, hid_dim)
            x_cross = x_cross.transpose(1, 2)
            # print("x_cross:", get_quantile(x_out.transpose(1, 2)[:, selected_channel, :], dim=2))
            # print("x_cross:", torch.mean(x_out[:, selected_channel, :], dim=2))
            # print("gate_mocha: ", gate_mocha[:, selected_channel, :])
            attn_mask_cross = None

        elif self.monotonic_approach == "hard_phoneme":
            #attn_mask_cross = x_mask.repeat(1, 1, r.shape[2]).transpose(1, 2)         # (b, tq, ts) <- ts is not masked
            #print("before diag:", attn_mask_cross)
            #attn_mask_cross = generate_diagal_fromMask(attn_mask_cross)
            attn_mask_cross = attnCross

        elif self.monotonic_approach == "mas":
            """
            attn_mask_cross = search_ma(x, r, x_mask)  # (b, tq, ts)
            """
            attn_mask_cross = attnCross
            print(attn_mask_cross[0, :10, :10])
        else:
            attn_mask_cross = x_mask.repeat(1, r.shape[2], 1).transpose(1, 2)

        if attn_mask_cross is not None:
            attn_mask_cross = torch.zeros_like(attn_mask_cross).masked_fill(attn_mask_cross == 0,
                                                                          -torch.finfo(x.dtype).max)  # 0(needtomask) -> infi, 1 -> 0

        r = self.rnorm(r.transpose(1, 2)).transpose(1, 2)  # (b, hid_dim, cut_l)
        x_cross, attn_map = self.attn2(x=x, c=r, attn_mask=attn_mask_cross)   # attn_map: (b)
        ############## CHECK01 ############
        #if not self.training:
        #    from GradTTS.utils import save_plot
        #    #print("attn_mask_cross -> block diag:", attn_mask_cross)
        #    save_plot(attn_mask_cross[0].cpu(), f'watchImg/attn_mask_cross.png')
        #    save_plot(attn_map[0][0].detach().cpu(), f'watchImg/attn_map.png')


        #print("attn_map", attn_map)
        x = x + gate_mocha * x_cross * x_mask  # b, hid_dim, cut_l

        # FFN
        x = x + gate_mlp * self.mlp(self.modulate(self.norm2(x.transpose(1, 2)).transpose(1, 2), shift_mlp, scale_mlp),
                                    x_mask)
        #print("x_out:", torch.mean(x[:, selected_channel, :], dim=2))
        # no condition version
        # x = x + self.attn(self.norm1(x.transpose(1,2)).transpose(1,2),  attn_mask)
        # x = x + self.mlp(self.norm2(x.transpose(1,2)).transpose(1,2), x_mask)
        #return x, stats["beta"]
        return x, attn_map    # only first head

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift


class DiTConVBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_channels, filter_channels, num_heads, kernel_size=3, p_dropout=0.1, gin_channels=0,
                 bAttn_type="base", **mochaAttn_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.attn = MultiHeadAttention(hidden_channels, hidden_channels, num_heads, p_dropout)
        self.norm2 = nn.LayerNorm(hidden_channels, elementwise_affine=False)

        self.mlp = FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(gin_channels, hidden_channels) if gin_channels != hidden_channels else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_channels, 6 * hidden_channels, bias=True)
        )

    def forward(self, x, c, x_mask=None):
        """
        Args:
            x : [batch_size, channel, time]
            c : [batch_size, channel]
            r: [batch_size, channel, time]
            x_mask : [batch_size, 1, time]
        return the same shape as x
        """
        x = x * x_mask
        attn_mask = x_mask.unsqueeze(1) * x_mask.unsqueeze(-1)  # shape: [batch_size, 1, time, time]
        attn_mask = torch.zeros_like(attn_mask).masked_fill(attn_mask == 0, -torch.finfo(x.dtype).max)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).unsqueeze(2).chunk(6,
                                                                                                                     dim=1)  # shape: [batch_size, channel, 1]
        x = x + gate_msa * self.attn(self.modulate(self.norm1(x.transpose(1, 2)).transpose(1, 2), shift_msa, scale_msa),
                                     attn_mask) * x_mask
        x = x + gate_mlp * self.mlp(self.modulate(self.norm2(x.transpose(1, 2)).transpose(1, 2), shift_mlp, scale_mlp),
                                    x_mask)
        # no condition version
        # x = x + self.attn(self.norm1(x.transpose(1,2)).transpose(1,2),  attn_mask)
        # x = x + self.mlp(self.norm2(x.transpose(1,2)).transpose(1,2), x_mask)
        return x

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift

class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)

