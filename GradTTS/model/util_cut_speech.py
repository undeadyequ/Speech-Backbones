import random

import torch

from GradTTS.model.util_mask_process import sequence_mask

def cutoff_inputs(y, y_lengths, y_mask, melstyle, melstyle_lengths, attn, cut_size, y_dim=80, q_seq_dur=None, k_seq_dur=None):
    if not isinstance(cut_size, type(None)):
        # randomly select start and end point
        max_offset = (y_lengths - cut_size).clamp(0)
        offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
        out_offset = torch.LongTensor([
            torch.tensor(random.choice(range(start, end)) if end > start else 0)
            for start, end in offset_ranges
        ]).to(y_lengths)

        # initiate tensor
        attn_cut = torch.zeros(attn.shape[0], attn.shape[1], cut_size, dtype=attn.dtype, device=attn.device)
        y_cut = torch.zeros(y.shape[0], y_dim, cut_size, dtype=y.dtype, device=y.device)
        melstyle_cut = torch.zeros(melstyle.shape[0], melstyle.shape[1], cut_size,
                                        dtype=melstyle.dtype, device=melstyle.device)
        y_cut_lengths = []
        melstyle_lengths = []

        if k_seq_dur is not None:
            q_seq_dur_cut = torch.zeros(q_seq_dur.shape[0], q_seq_dur.shape[1], cut_size,
                                        dtype=q_seq_dur.dtype, device=q_seq_dur.device)
            k_seq_dur_cut = torch.zeros(k_seq_dur.shape[0], k_seq_dur.shape[1], cut_size,
                                        dtype=k_seq_dur.dtype, device=k_seq_dur.device)
            # assign values
            for i, (y_, out_offset_, enc_hid_cond_, q_seq_dur_, k_seq_dur_) in enumerate(zip(y, out_offset, melstyle, q_seq_dur, k_seq_dur)):
                y_cut_length = cut_size + (y_lengths[i] - cut_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
                melstyle_lengths.append(y_cut_length)
                melstyle_cut[i, :, :y_cut_length] = enc_hid_cond_[:, cut_lower:cut_upper]
                q_seq_dur_cut[i, :, :y_cut_length] = torch.clamp(q_seq_dur_[:, cut_lower:cut_upper] - q_seq_dur_[:, cut_lower], min=0)
                k_seq_dur_cut[i, :, :y_cut_length] = torch.clamp(k_seq_dur_[:, cut_lower:cut_upper] - k_seq_dur_[:, cut_lower], min=0)

            q_seq_dur = q_seq_dur_cut
            k_seq_dur = k_seq_dur_cut
        else:
            # assign values
            for i, (y_, out_offset_, enc_hid_cond_) in enumerate(zip(y, out_offset, melstyle)):
                y_cut_length = cut_size + (y_lengths[i] - cut_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
                melstyle_lengths.append(y_cut_length)
                melstyle_cut[i, :, :y_cut_length] = enc_hid_cond_[:, cut_lower:cut_upper]
            q_seq_dur = None
            k_seq_dur = None

        y_cut_lengths = torch.LongTensor(y_cut_lengths)
        y_cut_mask = sequence_mask(y_cut_lengths, max_length=cut_size).unsqueeze(1).to(
            y_mask)  # Set length to cut_size to enable fix length learning
        melstyle_lengths = torch.LongTensor(melstyle_lengths)

        attn = attn_cut
        y = y_cut
        y_mask = y_cut_mask
        melstyle = melstyle_cut
        y_lengths = y_cut_lengths

    return y, y_lengths, y_mask, melstyle, melstyle_lengths, attn, q_seq_dur, k_seq_dur


def cutoff_inputs_old(y, y_lengths, y_mask, melstyle, melstyle_lengths, attn, cut_size, y_dim=80, q_seq_dur=None, k_seq_dur=None):
    """
    cut off inputs for larger batches
    Returns:
    """
    if not isinstance(cut_size, type(None)):
        # randomly select start and end point
        max_offset = (y_lengths - cut_size).clamp(0)
        offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
        out_offset = torch.LongTensor([
            torch.tensor(random.choice(range(start, end)) if end > start else 0)
            for start, end in offset_ranges
        ]).to(y_lengths)

        # initiate tensor
        attn_cut = torch.zeros(attn.shape[0], attn.shape[1], cut_size, dtype=attn.dtype, device=attn.device)
        y_cut = torch.zeros(y.shape[0], y_dim, cut_size, dtype=y.dtype, device=y.device)
        melstyle_cut = torch.zeros(melstyle.shape[0], melstyle.shape[1], cut_size,
                                        dtype=melstyle.dtype, device=melstyle.device)

        y_cut_lengths = []
        melstyle_lengths = []

        if k_seq_dur is not None:
            q_seq_dur_cut = torch.zeros(q_seq_dur.shape[0], q_seq_dur.shape[1], cut_size,
                                        dtype=q_seq_dur.dtype, device=q_seq_dur.device)
            k_seq_dur_cut = torch.zeros(k_seq_dur.shape[0], k_seq_dur.shape[1], cut_size,
                                        dtype=k_seq_dur.dtype, device=k_seq_dur.device)

            # assign values
            for i, (y_, out_offset_, enc_hid_cond_, q_seq_dur_, k_seq_dur_) in enumerate(zip(y, out_offset, melstyle, q_seq_dur, k_seq_dur)):
                y_cut_length = cut_size + (y_lengths[i] - cut_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
                melstyle_lengths.append(y_cut_length)
                melstyle_cut[i, :, :y_cut_length] = enc_hid_cond_[:, cut_lower:cut_upper]
                q_seq_dur_cut[i, :, :y_cut_length] = q_seq_dur_[:, cut_lower:cut_upper]
                k_seq_dur_cut[i, :, :y_cut_length] = k_seq_dur_[:, cut_lower:cut_upper]
            q_seq_dur = q_seq_dur_cut
            q_seq_dur = q_seq_dur - torch.min(q_seq_dur, -1)[0].unsqueeze(1)   # Let seq_dur start from 0, instead of cut point
            k_seq_dur = k_seq_dur_cut
            k_seq_dur = k_seq_dur - torch.min(k_seq_dur, -1)[0].unsqueeze(1)  # Let seq_dur start from 0, instead of cut point

        else:
            # assign values
            for i, (y_, out_offset_, enc_hid_cond_) in enumerate(zip(y, out_offset, melstyle)):
                y_cut_length = cut_size + (y_lengths[i] - cut_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
                melstyle_lengths.append(y_cut_length)
                melstyle_cut[i, :, :y_cut_length] = enc_hid_cond_[:, cut_lower:cut_upper]
            q_seq_dur = None
            k_seq_dur = None

        y_cut_lengths = torch.LongTensor(y_cut_lengths)
        y_cut_mask = sequence_mask(y_cut_lengths, max_length=cut_size).unsqueeze(1).to(
            y_mask)  # Set length to cut_size to enable fix length learning
        melstyle_lengths = torch.LongTensor(melstyle_lengths)

        attn = attn_cut
        y = y_cut
        y_mask = y_cut_mask
        melstyle = melstyle_cut
        y_lengths = y_cut_lengths

    return y, y_lengths, y_mask, melstyle, melstyle_lengths, attn, q_seq_dur, k_seq_dur


def cutoff_inputs_new(y, y_lengths, y_mask, melstyle, melstyle_lengths, attn, cut_size, y_dim=80, q_seq_dur=None, k_seq_dur=None):
    """
    cut y with given size, and adaptively cut melstyle with same phoneme size as cutted y.

    y: (b, y_length)
    q_seq_dur: (b, 1, mu_y_len)
    k_seq_durk_seq_dur: (b, 1, melstyle_len)
    cut off inputs for larger batches
    Returns:
        y:  ()
        melstyle:
    """
    if not isinstance(cut_size, type(None)):
        # randomly select start and end point
        max_offset = (y_lengths - cut_size).clamp(0)
        offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
        out_offset = torch.LongTensor([
            torch.tensor(random.choice(range(start, end)) if end > start else 0)
            for start, end in offset_ranges
        ]).to(y_lengths)

        # initiate tensor
        attn_cut = torch.zeros(attn.shape[0], attn.shape[1], cut_size, dtype=attn.dtype, device=attn.device)
        y_cut = torch.zeros(y.shape[0], y_dim, cut_size, dtype=y.dtype, device=y.device)

        melstyle_cut = torch.zeros(melstyle.shape[0], melstyle.shape[1], melstyle.shape[2], dtype=melstyle.dtype, device=melstyle.device)
        y_cut_lengths = []
        melstyle_cut_lengths = []

        if k_seq_dur is not None:
            q_seq_dur_cut = torch.zeros(q_seq_dur.shape[0], q_seq_dur.shape[1], cut_size,
                                        dtype=q_seq_dur.dtype, device=q_seq_dur.device)
            k_seq_dur_cut = torch.zeros(k_seq_dur.shape[0], k_seq_dur.shape[1], k_seq_dur.shape[2],
                                        dtype=k_seq_dur.dtype, device=k_seq_dur.device) # # k_seq has the same p_size (not f_size) with q_seq,
            # Sssign values
            melstyle_cut_len_max = 0
            for i, (y_, out_offset_, enc_hid_cond_, q_seq_dur_, k_seq_dur_) in enumerate(
                    zip(y, out_offset, melstyle, q_seq_dur, k_seq_dur)):
                # cut y related
                y_cut_length = cut_size + (y_lengths[i] - cut_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
                q_seq_dur_cut[i, 0, :y_cut_length] = reset_start_of_tensor(q_seq_dur_[0, cut_lower:cut_upper])
                q_seq_dur_cut[i, 0, :y_cut_length] = q_seq_dur_[0, cut_lower:cut_upper]

                # cut melstyle
                k_seq_dur_cut_, melsytyle_cut_lower, melsytyle_cut_upper = synfdur2reffdur(cut_lower, cut_upper, q_seq_dur_[0], k_seq_dur_[0])  # seq_dur is one channel

                k_seq_dur_cut_ = reset_start_of_tensor(k_seq_dur_cut_)

                #print("q_seq_dur_cut", q_seq_dur_cut[i])
                #print("k_seq_dur_cut_", k_seq_dur_cut_)
                k_seq_dur_cut_len_ = len(k_seq_dur_cut_.nonzero(as_tuple=True)[0])
                if k_seq_dur_cut_len_ > melstyle_cut_len_max:
                    melstyle_cut_len_max = k_seq_dur_cut_len_
                k_seq_dur_cut[i, 0, :k_seq_dur_cut_len_] = k_seq_dur_cut_[:k_seq_dur_cut_len_]

                melstyle_cut_lengths.append(k_seq_dur_cut_len_)
                melstyle_cut[i, :, :melsytyle_cut_upper - melsytyle_cut_lower] = enc_hid_cond_[:, melsytyle_cut_lower : melsytyle_cut_upper]
                # k_seq_dur_cut[i, :, :y_cut_length] = k_seq_dur_[:, cut_lower:cut_upper]

            # Get len of nonzero values of k_seq_dur
            k_seq_dur_cut = k_seq_dur_cut[:, :, :melstyle_cut_len_max]
            melstyle_cut = melstyle_cut[:, :, :melstyle_cut_len_max]

            q_seq_dur = q_seq_dur_cut
            k_seq_dur = k_seq_dur_cut

        else:
            # assign values
            for i, (y_, out_offset_, enc_hid_cond_) in enumerate(zip(y, out_offset, melstyle)):
                y_cut_length = cut_size + (y_lengths[i] - cut_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
                melstyle_cut_lengths.append(y_cut_length)
                melstyle_cut[i, :, :y_cut_length] = enc_hid_cond_[:, cut_lower:cut_upper]
            q_seq_dur = None
            k_seq_dur = None

        y_cut_lengths = torch.LongTensor(y_cut_lengths)
        y_cut_mask = sequence_mask(y_cut_lengths, max_length=cut_size).unsqueeze(1).to(
            y_mask)  # Set length to cut_size to enable fix length learning
        melstyle_cut_lengths = torch.LongTensor(melstyle_cut_lengths)

        attn = attn_cut
        y = y_cut
        y_mask = y_cut_mask
        melstyle = melstyle_cut
        y_lengths = y_cut_lengths
    return y, y_lengths, y_mask, melstyle, melstyle_cut_lengths, attn, q_seq_dur, k_seq_dur


def reset_start_of_tensor(tensor):
    """
    Reset start point of ordered tensor to 0.
    Examples:
        [5, 6, 8, 10, 0] -> [0, 1, 3, 5, 0]
    Args:
        tensor: (tensor_len, )
    Returns:
    """
    # get min vlaue
    """
    if tensor[0] == 0:
        return tensor
    else:
        for v in tensor:
            if v != 0:
                min_v = v
                break
    """
    return torch.clamp(tensor - tensor[0] + 1, min=0)


def synfdur2reffdur(syn_cut_flower, syn_cut_fupper, syn_fdur, ref_fdur):
    """
    Args:
        syn_cut_flower: int
        syn_cut_fupper: int
        syn_fdurs: (syn_f_len)
        ref_fdurs: (ref_f_len)
    Returns:
        ref_cut_fdurs: (cut_len)
    """
    syn_cut_plower, syn_cut_pupper = syn_fdur[syn_cut_flower], syn_fdur[syn_cut_fupper - 1]
    if syn_cut_plower == torch.max(ref_fdur):
        pass
        print("Not implement")
    ref_cut_flower = (ref_fdur == syn_cut_plower).nonzero(as_tuple=True)[0][0].int()  # 0=coord_1dim, 0=1st_match
    if syn_cut_pupper == 0 or syn_cut_pupper == torch.max(ref_fdur):  # [..., 0<-] or [..., 8<-(, 0)]
        ref_cut_fupper = len(ref_fdur)
    else:  # [..., 4<-, 5, ...]
        ref_cut_fupper = (ref_fdur == syn_cut_pupper + 1).nonzero(as_tuple=True)[0][
            0].int()  # 0=coord_1dim, 0=1st_match
    return ref_fdur[ref_cut_flower: ref_cut_fupper], ref_cut_flower, ref_cut_fupper


def synfdurs2reffdurs(syn_cut_flower, syn_cut_fupper, syn_fdurs, ref_fdurs):
    """
    Args:
        syn_cut_flower: int
        syn_cut_fupper: int
        syn_fdurs: (b, syn_f_len)
        ref_fdurs: (b, ref_f_len)
    Returns:
        ref_cut_fdurs: (b, cut_len)
    """

    ref_cut_fdurs = []
    syn_cut_plowers, syn_cut_puppers = syn_fdurs[:, syn_cut_flower], syn_fdurs[:, syn_cut_fupper - 1]
    for i, (syn_cut_plower, syn_cut_pupper) in enumerate(zip(syn_cut_plowers, syn_cut_puppers)):
        if syn_cut_plower == torch.max(ref_fdurs[i]):
            print("Not implement")
        ref_cut_flower = (ref_fdurs[i] == syn_cut_plower).nonzero(as_tuple=True)[0][
            0].int()  # 0=coord_1dim, 0=1st_match
        if syn_cut_pupper == 0 or syn_cut_pupper == torch.max(ref_fdurs[i]):  # [..., 0<-] or [..., 8<-(, 0)]
            ref_cut_fupper = len(ref_fdurs[i])
        else:  # [..., 4<-, 5, ...]
            ref_cut_fupper = (ref_fdurs[i] == syn_cut_pupper + 1).nonzero(as_tuple=True)[0][
                0].int()  # 0=coord_1dim, 0=1st_match
        ref_cut_fdur = ref_fdurs[i, ref_cut_flower: ref_cut_fupper]
        ref_cut_fdurs.append(ref_cut_fdur)

    pad_len = max([len(ref_cut_fdur) for ref_cut_fdur in ref_cut_fdurs])
    for i, ref_cut_fdur in enumerate(ref_cut_fdurs):
        if len(ref_cut_fdur) < pad_len:
            ref_cut_fdurs[i] = torch.cat(
                [ref_cut_fdur, torch.zeros(pad_len - len(ref_cut_fdur), dtype=ref_cut_fdur.dtype)])
        else:
            ref_cut_fdurs[i] = ref_cut_fdur[:pad_len]

    ref_cut_fdurs = torch.stack(ref_cut_fdurs, dim=0)
    return ref_cut_fdurs


def get_cut_range_x(attn, cut_lower, cut_upper):
    cut_lower_x = (attn[:, cut_lower] == 1).nonzero(as_tuple=False)
    cut_upper_x = (attn[:, cut_upper - 1] == 1).nonzero(as_tuple=False)
    if cut_upper_x.size()[0] == 0:
        torch.set_printoptions(threshold=10_000)
        # print(attn.size(), cut_lower, cut_upper)
        # print(attn[:, cut_upper])
        # print(attn)
    cut_lower_x = cut_lower_x[0][0]
    cut_upper_x = cut_upper_x[0][0]
    # print("cut_lower_x, cut_upper_x:{}, {}".format(cut_lower_x, cut_upper_x))
    return cut_lower_x, cut_upper_x


def get_first_last_nonZero(x2y_attn, x_index_start, x_index_end):
    """
    get y_index at x2y_attn[x_index, :] where is the first/last nonzero.
    Args:
        x2y_attn (1, 1, x_len, y_len):
        x_index_start:
        x_index_end:
    Returns:

    """
    if len(x2y_attn.shape) == 4 and x2y_attn.shape[0] == 1 and x2y_attn.shape[1] == 1:
        y_nonzero_index_start = (x2y_attn[0, 0, x_index_start, :] == 1).nonzero(as_tuple=True)
        y_nonzero_index_start = y_nonzero_index_start[0][0]  # 0: 3rd dim, 0: first nonzero
        y_nonzero_index_end = (x2y_attn[0, 0, x_index_end, :] == 1).nonzero(as_tuple=True)
        y_nonzero_index_end = y_nonzero_index_end[0][-1]  # 0: 3rd dim, 0: last nonzero
    else:
        raise IOError("x2y_attn should have 4 dims and 1 length for first two dims!")
    return y_nonzero_index_start, y_nonzero_index_end