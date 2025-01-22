
class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout

        self.k_channels = channels // n_heads
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)

        # from https://nn.labml.ai/transformers/rope/index.html
        self.query_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)
        self.key_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)

        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        x = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        query = self.query_rotary_pe(query)  # [b, n_head, t, c // n_head]
        key = self.key_rotary_pe(key)

        output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                                                dropout_p=self.p_dropout if self.training else 0)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output
