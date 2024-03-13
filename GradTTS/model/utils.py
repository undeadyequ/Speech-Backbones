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


def align(
        condition,
        align_len,
        align_att=None,
        condtype="noseq",
):
    """
    a: (b, dim, sr_len)
    tr_seq_len:
    attn: (b, sr_len, tr_len)

    Args:
        condition: (b, x_len/y_len, dim) or (b, dim)
        align_len: scalar
        align_att: (x_len, y_len)
        condtype:

    Returns:
        cond_aligned: (b, x_len/y_len, dim)
    """
    if condtype == "seq":
        if align_att is not None:
            cond_aligned = align_a2b(condition.transpose(1, 2),
                                     align_len,
                                     align_att.squeeze(1).transpose(1, 2))
        else:
            print("attn_mtx should not be NONE!!")
        return cond_aligned
    elif condtype == "noseq":
        cond_aligned = condition.unsqueeze(2).repeat(1, 1, align_len)   # (b, mel_dim, pmel_len)
    else:
        print("wrong condtype!")

    return cond_aligned


import numpy as np
import tgt
import librosa
import pyworld as pw


def get_alignment(tier, sampling_rate=16000, hop_length=256):
    """
    Get alignment from TextGrid file
    Args:
        tier:
        sampling_rate:
        hop_length:

    Returns:

    """
    sil_phones = ["sil", "sp", "spn"]

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            # For silent phones
            phones.append(p)

        durations.append(
            int(
                np.round(e * sampling_rate / hop_length)
                - np.round(s * sampling_rate / hop_length)
            )
        )

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]

    return phones, durations, start_time, end_time


def extract_pitch(wav_f,
                  wav_tg_f,
                  sampling_rate=16000,
                  hop_length=256,
                  need_trim=False
                  ):
    """

    Args:
        wav_f:
        sampling_rate:
        hop_length:

    Returns:

    """
    # get start end
    textgrid = tgt.io.read_textgrid(wav_tg_f)
    phone, duration, start, end = get_alignment(
        textgrid.get_tier_by_name("phones"),
        sampling_rate,
        hop_length
    )

    # get trimed wav
    wav, fs = librosa.load(wav_f)
    if need_trim:
        # trim by textgrid
        wav = wav[int(fs * start): int(fs * end)]

    wav = wav.astype(np.float64)

    # get pitch
    _f0, _time = pw.dio(wav,
                        fs,
                        frame_period=256 / sampling_rate * 1000
                        )
    f0 = pw.stonemask(wav, _f0, _time, fs)

    return f0