# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import torch
from einops import rearrange

from model.base import BaseModule


class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs)
        if type(output) is tuple:
            if len(output) == 2:
                return output[0] * self.g, output[1]
            elif len(output) == 3:
                return output[0] * self.g, output[1], output[2]
            else:
                IOError("Wrong output")
        else:
            return output * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                               dim_out))
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(BaseModule):
    """
    d_q, d_k, d_v = len_qkv * ft_dim
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)            

    def forward(self, x):
        b, c, h, w = x.shape  # b, c, dim, len
        qkv = self.to_qkv(x)  # b, dim_heads * 3 (qkv) * head, dim, len
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads=self.heads, qkv=3)  # 2, head, dim_heads, dim * len
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out), context


class MultiAttention2(BaseModule):
    def __init__(self, dim_in_q, dim_in_k, att_dim, heads=4):
        super(MultiAttention2, self).__init__()

        #hidden_dim = heads * att_dim
        self.W_Q = torch.nn.Conv2d(dim_in_q, att_dim * heads, 1, bias=False)
        # ?? W_k/W_V linear(dim_in, att_dim * head)
        self.W_K = torch.nn.Linear(dim_in_k, att_dim * heads)
        self.W_V = torch.nn.Linear(dim_in_k, att_dim * heads)

        #self.concat = torch.nn.Linear(att_dim, att_dim)
        self.to_out = torch.nn.Linear(att_dim * heads, dim_in_q)
        self.heads = heads
        self.att_dim = att_dim

    def forward(self, input_Q, key=None, value=None, attn_mask=None, mask_value=0):
        """

        Args:
            input_Q: (b, c, d_q, l_q)   c = dim_in_q
            key:    (b, d_k, l_k)       d_k = dim_in_k
            value:  (b, d_k, l_k)
            attn_mask: Mask attention score map for interpolation.
        Returns:
            out:  (b, c, d_q, l_q)
        """
        b, c, d_q, l_q = input_Q.shape
        d_k, l_k = key.shape[1], key.shape[2]
        # Change order of l_q and d_q for using mask
        input_Q = input_Q.view(b, c, l_q, d_q)  # <- BE CAREFUL the d_q, l_q order

        residual = input_Q.view(b, c, l_q * d_q).transpose(1, 2)  # (b, d_q * l_q, c)  ?? transpose works same as view ??
        Q = (
            self.W_Q(input_Q).view(b, self.heads, -1, l_q, d_q)
        )  #  (b, h, d_m, l_q, d_q)
        Q = Q.view(b, self.heads, self.att_dim,  l_q * d_q).transpose(2, 3) # -> (b, h, d_q * l_q, d_m)

        K = (
            self.W_K(key.transpose(1, 2)).view(b, self.heads, l_k, -1)
        )  # OUT: (b, h, l_k, d_m)
        V = (
            self.W_V(value.transpose(1, 2)).view(b, self.heads, l_k, -1)
        )  # OUT: (b, h, l_k, d_m)
        context, attn = ScaledDotProductionAttention()(Q, K, V, attn_mask, mask_value)   # (b, h, d_q * l_q, d_m)  attn: (b, h, l_q * d_q, l_k)
        context = torch.cat(
            [context[:, i, :, :] for i in range(context.size(1))], dim=-1)   # (b, d_q * l_q, d_m * h)  # why not use view

        output = self.to_out(context)    # -> (b, d_q * l_q, c)
        output = output + residual       # -> (b, d_q * l_q, c)  # There are 2 residual

        if attn_mask is not None:        # Mask on output+1st residual
            attn_mask_nohead = attn_mask[:, 0, :, :].squeeze(1)   # return this for mask on 2nd residual
            output = output.masked_fill(attn_mask_nohead, mask_value)
        else:
            attn_mask_nohead = None
        if output.get_device() == -1:
            output = torch.nn.LayerNorm(c)(output).transpose(1, 2)      # (b, c, d_q * l_q)
            output = output.view(b, c, l_q, d_q).transpose(2, 3)        # (b, c, d_q, l_q)
        else:
            output = torch.nn.LayerNorm(c).cuda()(output).transpose(1, 2)  # (b, c, d_q * l_q)
            output = output.view(b, c, l_q, d_q).transpose(2, 3)

        if attn_mask_nohead is not None:
            return output, attn, attn_mask_nohead
        else:
            return output, attn


class MultiAttention3(BaseModule):
    def __init__(self, c_q, d_k, att_dim, heads=4, origin_c_q=64):
        super(MultiAttention3, self).__init__()

        self.heads = heads
        self.att_dim = att_dim
        self.c_q = c_q
        self.d_k = d_k
        self.d_q = int(80 / int(c_q / origin_c_q))

        #hidden_dim = heads * att_dim
        self.W_Q = torch.nn.Conv2d(c_q, att_dim * heads, 1, bias=False)
        self.W_K = torch.nn.Linear(d_k, att_dim * heads * self.d_q)  #
        self.W_V = torch.nn.Linear(d_k, att_dim * heads * self.d_q)
        #self.W_K = torch.nn.Linear(d_k, att_dim * heads * self.d_q)  #
        #self.W_V = torch.nn.Linear(d_k, att_dim * heads * self.d_q)
        self.to_out = torch.nn.Linear(self.heads*self.att_dim, c_q)


    def forward(self, input_Q, key=None, value=None, attn_mask=None, mask_value=0):
        """
        frame2frame attention
        Args:
            input_Q: (b, c, d_q, l_q)   c = dim_in_q
            key:    (b, d_k, l_k)       d_k = d_k
            value:  (b, d_k, l_k)
            attn_mask: # [b, h, l_q, 1] Mask attention score map for interpolation.
        Returns:
            out:  (b, c, d_q, l_q)

        c = dim
        d_q: h
        att_dim: c
        l_q: w
        d_k: h1
        l_k: w1
        """
        b, c, d_q, l_q = input_Q.shape
        d_k, l_k = key.shape[1], key.shape[2]
        # Change order of l_q and d_q for using mask
        input_Q = input_Q.view(b, c, l_q, d_q)  # <- BE CAREFUL the d_q, l_q order

        #residual = input_Q.view(b, c * d_q, l_q).transpose(1, 2)  # (b, d_q * l_q, c)  ?? transpose works same as view ??
        # Dim(att) = target_frameN (or l_q) * ref_frameN (or l_k)
        Q = (
            self.W_Q(input_Q).view(b, self.heads, l_q, self.att_dim * d_q)
        )
        K = (
            self.W_K(key.transpose(1, 2)).view(b, self.heads, l_k, self.att_dim * d_q)
        )
        V = (
            self.W_V(value.transpose(1, 2)).view(b, self.heads, l_k, self.att_dim * d_q)
        ) # The last dim of Q, K, V should be same
        context, attn = ScaledDotProductionAttention()(Q, K, V, attn_mask, mask_value)   # (b, h, l_q, d_q * self.att_dim)  attn: (b, h, l_q, l_k)
        #context = torch.cat(
        #    [context[:, i, :, :] for i in range(context.size(1))], dim=-1)   # (b, d_q * l_q, d_m * h)  # why not use view
        context = context.view(b, l_q, d_q, self.heads*self.att_dim)

        output = self.to_out(context)  # -> (b, l_q, d_q, c)
        resnet = input_Q.view(b, l_q, d_q, c)
        output = output + resnet       # -> (b, l_q, d_q, c)  # There is another mask outside in Resnet(Rezero(Attn()))


        if output.get_device() == -1:
            output = torch.nn.LayerNorm(c)(output)
            output = output.view(b, c, l_q, d_q).transpose(2, 3)  # (b, c, d_q, l_q)
        else:
            output = torch.nn.LayerNorm(c).cuda()(output)
            output = output.view(b, c, d_q, l_q)

        if attn_mask is not None:      # Mask on output + 1st residual
            #attn_mask_nohead = attn_mask[:, 0, :, :].squeeze(1)   # return this for mask on 2nd residual
            # convert mask (for f2f)
            attn_mask_nohead = attn_mask[:, 0, :, :].squeeze(1).transpose(1, 2).unsqueeze(2)  # return this for mask on 2nd residual
            output = output.masked_fill(attn_mask_nohead, mask_value)
        else:
            attn_mask_nohead = None

        if attn_mask_nohead is not None:
            #print(output)
            return output, attn, attn_mask_nohead   # return mask for 2nd residual
        else:
            return output, attn

class ScaledDotProductionAttention(BaseModule):
    def __init__(self, dropout: float = None, scale: bool = True):
        super(ScaledDotProductionAttention, self).__init__()
        if dropout is not None:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = torch.nn.Softmax(dim=-1)   #?? torch.nn.Softmax(dim=2) -> torch.nn.Softmax(dim=-1)
        self.scale = scale

    def forward(self, q, k, v, att_mask=None, mask_value=0):
        """
        Args:
            q: (b, h, q, d)
            k: (b, h, k, d)
            v: (b, h, k, d)
            att_mask: (b, h, q, k)
        Returns:
            output: (b, h, q, d)
            attn: (b, h, q, k)
        """
        attn = torch.einsum('bhqd,bhkd->bhqk', q, k)
        if self.scale:
            dimension = torch.as_tensor(k.size(-1), dtype=attn.dtype, device=attn.device).sqrt()
            attn = attn / dimension
        attn = self.softmax(attn)
        if att_mask is not None:
            attn = attn.masked_fill(att_mask, mask_value)  # Do mask after softmax
        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.einsum('bhqk,bhkd->bhqd', attn, v)
        return output, attn


class MultiAttention(BaseModule):
    def __init__(self, dim_in, att_dim=80):
        super(MultiAttention, self).__init__()
        self.dim_in = dim_in
        self.block1 = torch.nn.Sequential(torch.nn.Conv2d(dim_in, 1, 3,
                                         padding=1), Mish())  # 4d -> 3d
        #self.block2 = torch.nn.Sequential(torch.nn.Conv2d(1, dim_in, 1, padding=1))
        self.prj = torch.nn.Linear(240, att_dim)   # dim_kv -> dim_q
        self.mlthead = torch.nn.MultiheadAttention(embed_dim=att_dim, num_heads=2, batch_first=True)
    def forward(self, x, key, value):
        x = self.block1(x)
        x = x.squeeze(1).transpose(1, 2)   # (b, c, d, l_q) -> # (b, d, l_q)
        key = self.prj(key.transpose(1, 2))
        value = self.prj(value.transpose(1, 2)) # (b, 240, l_kv) -> # (b, 80, l_kv)
        #x = self.mlthead(x.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
        x = self.mlthead(x, key, value)   # (b, d, l_q)

        x_res = x[0].unsqueeze(1).repeat(1, self.dim_in, 1, 1)
        return x_res.transpose(2, 3)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn
        
    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs)
        if type(output) is tuple and len(output) >= 2:
            if len(output) == 3:
                attn_mask = output[2]
                #b, c, m, l = output[0].shape
                #attn_mask = attn_mask.view(b, -1, l, m).transpose(2, 3)  # ?? ad hoc
                #attn_mask = attn_mask.unsqueeze(2).transpose(1, 3)
                x = x.masked_fill(attn_mask, 0)
            return output[0] + x, output[1]
        else:
            return output + x

        
class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GradLogPEstimator2d(BaseModule):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8,
                 n_spks=None, spk_emb_dim=64, n_feats=80, pe_scale=1000):
        super(GradLogPEstimator2d, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.n_spks = n_spks if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale
        
        if n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4), Mish(),
                                               torch.nn.Linear(spk_emb_dim * 4, n_feats))
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(),
                                       torch.nn.Linear(dim * 4, dim))

        dims = [2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                     ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                     Residual(Rezero(LinearAttention(dim_in))),
                     Upsample(dim_in)]))
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask, mu, t, spk=None):
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)
        
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if self.n_spks < 2:
            x = torch.stack([mu, x], 1)
        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.stack([mu, x, s], 1)
        mask = mask.unsqueeze(1)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            # added to avoid size not matching
            hid_pop = hiddens.pop()
            hid_pop_size = hid_pop.size()
            if hid_pop_size[2] != x.size()[2] or hid_pop_size[3] != x.size()[3]:
                print("change old x shape {} to new one {} on last 2 dims".format(x.size(), hid_pop.size()))
                x = x[:, :, :hid_pop.size()[2], :hid_pop.size()[3]]
            #x = torch.cat((x, hiddens.pop()), dim=1)
            x = torch.cat((x, hid_pop), dim=1)

            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        if x.size()[3] != mask.size()[3]:
            x = x[:, :, :, :mask.size()[3]]
        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)


def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
    else:
        noise = beta_init + (beta_term - beta_init)*t
    return noise


class Diffusion(BaseModule):
    def __init__(self, n_feats, dim,
                 n_spks=1, spk_emb_dim=64,
                 beta_min=0.05, beta_max=20, pe_scale=1000):
        super(Diffusion, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        
        self.estimator = GradLogPEstimator2d(dim, n_spks=n_spks,
                                             spk_emb_dim=spk_emb_dim,
                                             pe_scale=pe_scale)

    def forward_diffusion(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        #print(xt.size(), z.size(), mask.size())
        # torch.Size([4, 80, 188]) torch.Size([4, 80, 188]) torch.Size([4, 1, 132])
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, 
                                cumulative=False)
            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - self.estimator(xt, mask, mu, t, spk)
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t, spk))
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def reverse_diffusion_interpolate(self, z, mask, mu, n_timesteps, stoc=False, spk1=None, spk2=None):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype,
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max,
                                cumulative=False)
            estm1 = self.estimator(xt, mask, mu, t, spk1)
            estm2 = self.estimator(xt, mask, mu, t, spk2)
            inter_estm12 = (estm1 + estm2) * 0.5

            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - inter_estm12
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - inter_estm12)
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, spk)

    # used for interpolation
    #@torch.no_grad()
    #def forward(self, z, mask, mu, n_timesteps, stoc=False, spk1=None, spk2=None):
    #    return self.reverse_diffusion_interpolate(z, mask, mu, n_timesteps, stoc, spk1, spk2)

    def loss_t(self, x0, mask, mu, t, spk=None):
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        noise_estimation = self.estimator(xt, mask, mu, t, spk)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z)**2) / (torch.sum(mask)*self.n_feats)
        return loss, xt

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mu, t, spk)