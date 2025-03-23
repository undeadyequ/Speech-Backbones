# References:
# https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/transformer.py
# https://github.com/jaywalnut310/vits/blob/main/attentions.py
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py

import torch
import torch.nn as nn
from GradTTS.model.estimator.stditBlockAttention3 import MultiHeadAttention, MultiHeadAttentionCross

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
    def __init__(self, hidden_channels, filter_channels, num_heads, kernel_size=3, p_dropout=0.1, gin_channels=0,
                 phoneme_RoPE=False):
        super().__init__()
        # self
        self.norm1 = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.attn = MultiHeadAttention(hidden_channels, hidden_channels, num_heads, p_dropout)
        self.norm2 = nn.LayerNorm(hidden_channels, elementwise_affine=False)

        # cross
        self.norm3 = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.attn2 = MultiHeadAttentionCross(hidden_channels, hidden_channels, num_heads, p_dropout, phoneme_RoPE)
        self.rnorm = nn.LayerNorm(hidden_channels, elementwise_affine=False)

        # FFN
        self.mlp = FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(gin_channels, hidden_channels) if gin_channels != hidden_channels else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_channels, 7 * hidden_channels, bias=True)
        )

    def forward(self, x, c, r=None, x_mask=None, muy_dur=None, ref_dur=None, attnCross=None, seq_dur=None):
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
        x_cross, attn_map = self.attn2(x=x, c=r, attn_mask=attn_mask_cross, seq_dur=seq_dur)   # attn_map: (b)

        x = x + gate_mocha * x_cross * x_mask  # b, hid_dim, cut_l

        # FFN
        x = x + gate_mlp * self.mlp(self.modulate(self.norm2(x.transpose(1, 2)).transpose(1, 2), shift_mlp, scale_mlp),
                                    x_mask)
        return x, attn_map

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

