""" from https://github.com/jaywalnut310/glow-tts """

import torch
import torch.nn.functional as F
import numpy as np
import tgt
import librosa
import pyworld as pw
from GradTTS.model import monotonic_align
import math


###################### Mask Related #############################

def sequence_dur(durs, seq_len):
    """
    sequence duration
    dur:     (b, mu_x_len)
    seq_len: (b, y_len)
    Returns
        dur_seq: (b, 1, y_len)
    Examples:
        # value = Index of phoneme,    times = nums of frame in this phoenme
        [[1, 3, 2], [2, 4, 1]]  ->
        [1, 2, 2, 2, 3, 3, 0], [1, 1, 2, 2, 2, 2, 3]]
    """
    durs_seq = torch.zeros_like(seq_len, device=durs.device)
    for i, dur in enumerate(durs.int()):
        dur_index = torch.arange(len(dur)).to(durs.device)   # dur
        dur_repeat = torch.repeat_interleave(dur_index, dur)  # repeat each element of dur by dur times
        durs_seq[i, :len(dur_repeat)] = dur_repeat
    return durs_seq



def sequence_mask(length, max_length=None):
    """
    sequence mask
    length: (b, 1)  -> (b, max(length))
    """
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


def get_diag_from_dur2(durs, max_len=100000):
    """
    get diag_dur from dur
    Args:
        dur: (b, frame_n)  # [[0, 0, 0, 1, 1, 2, ..., 0, 0], [3, 3, 5, 5, ... 0, 0]]  # not nessasary start from 0, inconsisitent
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
    p_l_all = int(torch.max(durs)) + 1   # p_l_all should not be normalized
    diag_durs = torch.zeros([b, p_l_all, frame_n]).to(device=durs.device)

    for b in range(durs_norm.shape[0]):
        dur = durs_norm[b]
        p_l = int(torch.max(durs_norm)) + 1   # phoneme nums
        for i in range(p_l):  # loop dim1
            search_pFrameLen = torch.where(dur == i)[0]
            if len(search_pFrameLen) != 0:
                try:
                    idxs = torch.where(dur == i)[0] if i != 0 else torch.arange(torch.nonzero(dur)[0][0])
                except:
                    print("1. i, dur, searchpFrameLen, indxs", i, dur, search_pFrameLen, idxs)
                    #print("error diag dur", idxs, frame_n)
                try:
                    diag_durs[b, i, idxs[0]:idxs[-1] + 1] = 1
                except:
                    print("2. i, dur, searchpFrameLen, indxs", i, dur, search_pFrameLen, idxs)
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
        diag_dur[i, start : start + d] = 1
        start = start + d
    return diag_dur

###################### Loss Related #############################

def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_)**2) / torch.sum(lengths)
    return loss


def mon_attention_loss(attn_w, mon_attention_mask):
    """
    Monotonic loss
    attn_w:
    mattention_mask: monotonic attention mask
    """
    mon_loss = torch.sum(torch.masked_fill(attn_w, mon_attention_mask))
    return mon_loss

def vae_loss():
    pass



###################### Attention Related #############################
def create_mon_attn_mask(attn_uy, attn_re, chunk_size=16):
    """
    args:
        attn_uy: (b, h, mux_len, muy_len)
        attn_ref: (b, h, rex_len, rey_len)
    return:
        mon_attn_mask: 
    """
    ub, uh, mux_len, muy_len = attn_uy.shape()
    rb, rh, rex_len, rey_len = attn_re.shape()
    
    if ub != rb or uh != rh:
        raise IOError("b {}{}, h {}{} should be same!".format(ub, rb, uh, rh))
    
    if rex_len != mux_len:
        transfer_len = min(rex_len, mux_len)
        attn_uy = attn_uy[:, :transfer_len, :, :]
        attn_re = attn_re[:, :transfer_len, :, :]
    mon_attn_mask = torch.einsum('bhyx,bhxy->bhyy', attn_uy.T(), attn_re)
    return mon_attn_mask


def search_ma(x, y, attn_mask, xy_dim=80):
    """
    Monotonic Alignment Search
    Returns:

    """
    const = -0.5 * math.log(2 * math.pi) * xy_dim
    factor = -0.5 * torch.ones(x.shape, dtype=x.dtype, device=x.device)
    y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
    y_mu_double = torch.matmul(2.0 * (factor * x).transpose(1, 2), y)
    mu_square = torch.sum(factor * (x ** 2), 1).unsqueeze(-1)
    log_prior = y_square - y_mu_double + mu_square + const
    attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
    return attn


###################### Align Related #############################

def align_a2b(a, tr_seq_len, attn=None):
    """
    align tensor a to b (tr_seq_len) by attn, if attn with wrong sr_len or tr_len, then
    1. padding when sr_len < tr_len
    2. cutting when sr_len > tr_len
    Args:
        a: (b, dim, sr_len)
        tr_seq_len:
        attn: (b, sr_len, tr_len)
    Returns:
        b: (b, dim, tr_seq_len)
    """
    # Used in training ?
    if attn.shape[1] == a.shape[2] and attn.shape[2] == tr_seq_len:
        b = torch.matmul(a, attn)
    else:
        sr_seq_len = a.shape[2]
        if tr_seq_len > sr_seq_len:
            left_pad = int((tr_seq_len - sr_seq_len) / 2)
            right_pad = tr_seq_len - left_pad - sr_seq_len
            p1d = (left_pad, right_pad)
            # b = a.permute(0, 2, 1)
            b = F.pad(a, p1d, "constant", 0)  # (b, word_len, 80) -> # (b, mel_len, 80)
            # b = b.permute(0, 2, 1)
            # psd = psd.unsqueeze(1)  # (b, 1, mel_len, 80)
        else:
            b = a[:, :, :tr_seq_len]
    return b


def align_a2b_padcut(a, tr_seq_len, pad_value=0, pad_type="two"):
    """
    padcut last dim of tensor to target length
    Args:
        a (b, d, l):
        tr_seq_len: int
        pad_value:
        pad_type:
    Returns:

    """
    sr_seq_len = a.shape[-1]
    if tr_seq_len > sr_seq_len:
        left_pad = int((tr_seq_len - sr_seq_len) / 2) if pad_type == "two" else 0
        right_pad = tr_seq_len - left_pad - sr_seq_len
        p1d = (left_pad, right_pad)
        b = F.pad(a, p1d, "constant", pad_value)  # (b, word_len, 80) -> # (b, mel_len, 80)
    else:
        b = a[..., :tr_seq_len]
    return b


def cut_pad_start_end(ref_start, ref_end, sr_seq_len, tr_seq_len):
    if tr_seq_len > sr_seq_len:
        left_pad = int((tr_seq_len - sr_seq_len) / 2)
        ref_start += left_pad
        ref_end += left_pad
    else:
        ref_end = min(ref_end, tr_seq_len)
    return ref_start, ref_end


def align(
        condition,
        align_len,
        align_att=None,
        condtype="noseq"
):
    """
    Align sequential or scala condition given align_len
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
            cond_aligned = align_a2b(
                condition.transpose(1, 2),
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
                  wav_tg_f=None,
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


    # get trimed wav
    wav, fs = librosa.load(wav_f)
    if need_trim:
        if wav_tg_f is not None:
            # get start end
            textgrid = tgt.io.read_textgrid(wav_tg_f)
            phone, duration, start, end = get_alignment(
                textgrid.get_tier_by_name("phones"),
                sampling_rate,
                hop_length
            )
        else:
            raise IOError("wav_tg_f should not be none!")
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


class pitchExtracdtion:
    def __init__(self):
        pass

    def extractPitch(self,
                     wav_f,
                     wav_tg_f,
                     sampling_rate=16000,
                     hop_length=256,
                     need_trim=False):
        extract_pitch(
            wav_f,
            wav_tg_f,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            need_trim=need_trim
        )
        
        
def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

    """
    return ~make_pad_mask(lengths, xs, length_dim)



def make_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


if __name__ == '__main__':
    attn_maps = torch.tensor([[[1, 1, 0]], [1, 1, 1], [1, 1, 0]])
    diags_attn_mpas = generate_diagal_fromMask(attn_maps)
    print("previous attn_maps", attn_maps)
    print("after attn_maps", diags_attn_mpas)