""" from https://github.com/jaywalnut310/glow-tts """

import torch
import torch.nn.functional as F

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], 
                                          [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_)**2) / torch.sum(lengths)
    return loss


def align_a2b(a, tr_seq_len, attn):
    """
    align tensor a to b (tr_seq_len) by attn, if attn with wrong sr_len or tr_len, then
    1. padding when sr_len < tr_len
    2. cutting when sr_len > tr_len
    Args:
        a: (b, dim, sr_len)
        tr_seq_len:
        attn: (b, sr_len, tr_len)
    Returns:

    """
    # check whether length of x is same with psd
    if attn.shape[1] == a.shape[2] and attn.shape[2] == tr_seq_len:
        b = torch.matmul(a, attn)
    else:
        sr_seq_len = a.shape[2]
        if tr_seq_len > sr_seq_len:
            left_pad = int((tr_seq_len - sr_seq_len) / 2)
            right_pad = tr_seq_len - left_pad - sr_seq_len
            p1d = (left_pad, right_pad)
            #b = a.permute(0, 2, 1)
            b = F.pad(a, p1d, "constant", 0)  # (b, word_len, 80) -> # (b, mel_len, 80)
            #b = b.permute(0, 2, 1)
            # psd = psd.unsqueeze(1)  # (b, 1, mel_len, 80)
        else:
            b = a[:, :, :tr_seq_len]
    return b