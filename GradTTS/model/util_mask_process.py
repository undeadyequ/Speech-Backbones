import torch
import torch.nn.functional as F
import numpy as np
import tgt
import librosa
import pyworld as pw
from GradTTS.model import monotonic_align
import math

###################### Mask Related #############################
"""
- Variable
p_dur:   (b, p_len)    # [2, 2, 1]
s_dur:   (b, s_len)    # [4, 1]      (Sum same as above)
f_dur:   (b, f_len)    # [1, 1, 2, 2, 3]   (=seq_dur)
diag_dur: (b, )

- Function
pdur2sdur
pdur2fdur
pdur2diag_dur
scale_dur
"""


def pdur2fdur(durs, seq_len, syl_start=None):
    """
    Sequence p-level duration to f-level duration in increase order
    durs:     (b, mu_x_len)
    seq_len: (b, y_len)
    syl_start: (b, sel_len)  [[0, 2], ..., []]
    Returns
        dur_seq: (b, 1, y_len)
    Examples:
        # value = Index of phoneme,    times = nums of frame in this phoenme
        [[1, 3, 2], [2, 4, 1]], [[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]]  ->
        [1, 2, 2, 2, 3, 3, 0], [1, 1, 2, 2, 2, 2, 3]]
    """
    # translate phone durs to syllable durs if needed
    if syl_start is not None:
        durs = pdur2syldur(durs, syl_start)

    durs_seq = torch.zeros_like(seq_len, device=durs.device)
    for i, dur in enumerate(durs.int()):
        #dur_index = torch.arange(1, len(dur) + 1).to(durs.device)  # [1, 3, 2] -> [1, 2, 3]
        dur_index = torch.arange(len(dur)).to(durs.device)  # [1, 3, 2] -> [1, 2, 3]

        dur_repeat = torch.repeat_interleave(dur_index, dur)  # repeat each element of dur by dur times exp: [1, 2, 3] -> [1, 2, 2, 2, 3, 3]
        if len(dur_repeat) <= durs_seq.shape[1]:
            durs_seq[i, :len(dur_repeat)] = dur_repeat
        else:  # it will not be happed?
            durs_seq[i, :] = dur_repeat[:durs_seq.shape[1]]
    return durs_seq


def pdur2syldur(phone_durs, syl_start_index):
    """
    phone_durs: (b, durlen)
    syl_start_index: (b, syl_len)
    """

    syl_durs = []
    for b, (p_dur, s_start) in enumerate(zip(phone_durs, syl_start_index)):
        syl_dur = []
        phone_len = sum(p_dur)
        # get syllables and its duration
        for i in range(len(s_start)):
            syl_start = s_start[i]
            if i != len(s_start) - 1:
                syl_end = s_start[i + 1]
            else:
                syl_end = phone_len  # last syllables includes to the last phone
            if isinstance(syl_start, torch.Tensor):
                syl_start = syl_start.item()
                syl_end = int(syl_end.item())
            syl_dur.append(sum(p_dur[syl_start: syl_end]))
        syl_durs.append(syl_dur)
    syl_durs = torch.Tensor(syl_durs).cuda()
    return syl_durs


def scale_dur(ref_dur, target_seq_len):
    """
    Extract f-level duration by scaling p-level duration of reference represents (same audio with target)
    to the same size of target sequence

    Args:
        target_seq: (b, melstyle_len)
        ref_dur:   (b, mu_x_len)
    Returns:
        target_dur: (b, 1, melstyle_len)
    Examples:
        (1, 2, 2), (15)  -> (3, 6, 6)
    """
    ref_dur_len = torch.sum(ref_dur, dim=1) * 1.0  # (b, )
    target_dur_len = target_seq_len * 1.0  # (b, )

    # scaling from ref_dur
    target_p_dur = (torch.transpose(ref_dur, 0, 1) * target_dur_len / ref_dur_len).to(torch.long)  # (mu_x_len, b)
    target_p_dur = torch.transpose(target_p_dur, 0, 1)  # (b, mu_x_len)
    target_p_dur[target_p_dur < 1] = 1  # set min to 1

    # handle with mismatch of f-level dur length of summing p-level dur and given target_seq_len
    for b in range(len(target_p_dur)):
        if torch.sum(target_p_dur[b]) != target_dur_len[b]:
            diffs = target_seq_len[b].long() - torch.sum(target_p_dur[b])
            target_p_dur[b][-1] += diffs  # add different to last phoneme
            target_p_dur[b][target_p_dur[b] < 0] = 0

    # target_seq = sequence_mask(target_seq_len).long()
    # target_dur_seq = sequence_dur(target_p_dur, target_seq)
    return target_p_dur


def scale_dur_new(ref_dur, target_seq_len):
    """
    Extract f-level duration by scaling p-level duration of reference represents (same audio with target)
    to the same size of target sequence

    Args:
        target_seq: (b, melstyle_len)
        ref_dur:   (b, mu_x_len)
    Returns:
        target_dur: (b, 1, melstyle_len)
    Examples:
        (1, 2, 2), (15)  -> (3, 6, 6)
    """
    ref_dur_len = torch.sum(ref_dur, dim=1) * 1.0  # (b, )
    target_dur_len = target_seq_len * 1.0  # (b, )

    # scaling from ref_dur
    target_p_dur = (torch.transpose(ref_dur, 0, 1) * target_dur_len / ref_dur_len).to(torch.long)  # (mu_x_len, b)
    target_p_dur = torch.transpose(target_p_dur, 0, 1)  # (b, mu_x_len)
    target_p_dur[(target_p_dur < 1) & (target_p_dur > 0)] = 1  # set min to 1

    # handle with mismatch of f-level dur length of summing p-level dur and given target_seq_len
    for b in range(len(target_p_dur)):
        if torch.sum(target_p_dur[b]) != target_dur_len[b]:
            diffs = target_seq_len[b].long() - torch.sum(target_p_dur[b])
            last_phone = target_p_dur[b].nonzero()[-1, 0]
            if diffs < 0:
                print("Not implement for downScale!!")
            target_p_dur[b][last_phone] += diffs  # add different to last phoneme
            # target_p_dur[b][target_p_dur[b] < 0] = 0

    # target_seq = sequence_mask(target_seq_len).long()
    # target_dur_seq = sequence_dur(target_p_dur, target_seq)
    return target_p_dur


def sequence_mask(length, max_length=None):
    """
    sequence mask
    length: (b, 1)  -> (b, max(length))
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


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


def generate_diagal_fromMask(y_lengths):
    """
    generate diagal Given mask
    args:
        y_lengths: (b, ) or # (b, tq, ts)
    Returns:
    """
    if len(y_lengths.shape) == 1:
        b, min_len = len(y_lengths), int(torch.max(y_lengths))
        diags = torch.zeros([b, min_len, min_len]).cuda()
        for i in range(b):
            diag = torch.diag(y_lengths[i])
            diags[i] = diag
    elif len(y_lengths.shape) == 3:
        b, tq, ts = y_lengths.shape
        diags = torch.zeros([b, tq, ts]).cuda()
        for i in range(b):
            non_zero_res = (y_lengths[i] == 1).nonzero(as_tuple=True)
            min_len = torch.min(non_zero_res[0][-1], non_zero_res[1][-1])
            diags[i][:min_len, :min_len] = torch.diag(torch.ones(min_len))
    else:
        IOError("only supoort dim 1 or 3")
    return diags


def generate_gd_diagMask(dur1, dur2, gran="phoneme"):
    """
    Generate GD phoneme2phoneme (or frame2frame) diagnal mask (b, tq, ts)
    , where dim1 = noise (query), dim2 = style (key, value)
    Args:
        dur1 (d_n, ): duration of phoneme/frame (d_n = p_L)
        dur2 (d_n, ):
    Returns:
        diagMask (b, tq, ts), where tq = sum(dur1), ts = sum(dur2)
    """
    # check dur_n
    if len(dur1) != len(dur2):
        print("dur1 and dur2 should be same!")

    dur1_diag = get_diag_from_dur(dur1)  # (d_n, t_q)
    dur2_diag = get_diag_from_dur(dur2)  # (d_n, t_s)

    return torch.matmul(dur1_diag, dur2_diag)


def create_p2pmask_from_seqdur(q_seq_dur, k_seq_dur):
    """
    Args:
        q_seq_dur: (b, mu_y_len)
        k_seq_dur: (b, melstyle_len)

    Example:
        (1, 1, 2, 2, 3), (2, 2, 2, 3, 3, 4)

        [[0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0]]
    Returns:
        (b, mu_y_len, melstyle_len)
    """
    diag_q_seq_dur = get_diag_from_dur2(q_seq_dur)  # (b, p_len, mu_y_len)
    diag_k_seq_dur = get_diag_from_dur2(k_seq_dur)  # (b, p_len, melstyle_len)

    if diag_q_seq_dur.shape[1] != diag_k_seq_dur.shape[1]:
        diag_q_seq_dur = diag_q_seq_dur[:, :diag_k_seq_dur.shape[1], :]   # donot attend the last n phonemes
        # print("q_seq, diag_q_seq, phone_num of q_seq", q_seq_dur.shape, diag_q_seq_dur.shape, torch.max(q_seq_dur, dim=1))
        # print("k_seq, diag_k_seq, phone_num of k_seq", k_seq_dur.shape, diag_k_seq_dur.shape, torch.max(k_seq_dur, dim=1))
        #raise IOError(
        #    "phone_len of k {} and q {} should be same".format(diag_q_seq_dur.shape[1], diag_k_seq_dur.shape[1]))


    return torch.matmul(diag_q_seq_dur.transpose(1, 2), diag_k_seq_dur)


def get_diag_from_dur2(durs, max_len=100000):
    """
    Get diagonal matrix from seq_dur (start from any numbers)
    Args:
        dur: (b, frame_n)  # [3, 3, 5, 5, ... 0, 0]]  # not nessasary start from 0, inconsisitent
        max_len: max len of duration  (becuase dur is not cut)
    Returns:
        diag_dur: (b, p_l, m_l)
        111000000000
        000110000000
        000001000000
    """
    #  Normalize each sample to starting from 0
    durs_norm = durs.clone()
    for b in range(durs.shape[0]):
        durs_norm[b] -= durs[b][0]

    b, frame_n = durs_norm.shape
    p_l_all = int(torch.max(durs)) + 1  # p_l_all should not be normalized
    diag_durs = torch.zeros([b, p_l_all, frame_n]).to(device=durs.device)

    for b in range(durs_norm.shape[0]):
        dur = durs_norm[b]
        p_l = int(torch.max(durs_norm)) + 1  # phoneme nums
        for i in range(p_l):  # loop dim1
            search_pFrameLen = torch.where(dur == i)[0]
            if len(search_pFrameLen) != 0:
                try:
                    idxs = torch.where(dur == i)[0] if i != 0 else torch.arange(torch.nonzero(dur)[0][0])
                except:
                    pass
                    # print("1. i, dur, searchpFrameLen, indxs", i, dur, search_pFrameLen, idxs)
                    # print("error diag dur", idxs, frame_n)
                try:
                    diag_durs[b, i, idxs[0]:idxs[-1] + 1] = 1
                except:
                    pass
                    # print("2. i, dur, searchpFrameLen, indxs", i, dur, search_pFrameLen, idxs)
    return diag_durs


def get_diag_from_dur(dur, max_len=100000):
    """
    Args:
        dur: (d_n, )   # [4, 2, 3]
        max_len: max len of duration  (becuase dur is not cut)
    Returns:
        diag_dur: (d_n, m_l)
    """
    diag_dur = torch.zeros([len(dur), min(torch.sum(dur), max_len)])
    start = 0
    for i, d in enumerate(dur):
        end = start + d
        if end > max_len:
            diag_dur[i, start: max_len] = 1
            break
        diag_dur[i, start: start + d] = 1
        start = start + d
    return diag_dur