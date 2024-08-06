import os.path
from einops import rearrange
import math
import torch
import torch.nn.functional as F
import numpy as np
import librosa
from GradTTS.model.utils import extract_pitch
from GradTTS.model.base import BaseModule
from GradTTS.model.diffusion import Downsample, Upsample
from GradTTS.model.utils import align
import matplotlib.pyplot as plt
from bisect import bisect
from pathlib import Path



class DiffAttnMask:
    def __init__(self):
        self.down_mid_up_ration = {
            "down": [(1, 1), (0.5, 0.5), (0.25, 0.25)],  # (c_ratio, lqk_ratio)
            "mid": [(0.25, 0.25)],
            "up": [(0.25, 0.25), (0.5, 0.25), (1, 0.5)],
        }
        self.in_out = [(1, 1), (1, 1), (1, 1)]
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])

        # Set unet
        """
        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.functional.interpolate(scale_factor=0.5, mode="nearest") if not is_last else torch.nn.Identity())
        for ind, (dim_in, dim_out) in enumerate(reversed(self.in_out[1:])):
            self.ups.append(
                torch.nn.functional.interpolate(scale_factor=2, mode="nearest")
            )
        """

    def create_p2p_mask3(self,
                         tgt_size: int = None,
                         ref_size: int = None,
                         mask_dims=4,
                         tgt_range=(10, 20),
                         ref_range=(10, 20)
                         ):
        """


        """
        p2p_masks_dict = {"down": [], "mid": [], "up": []}

        if tgt_range[1] > tgt_size or ref_range[1] > ref_size:
            print("Given tgt_range: {} or ref_range: {} is out of size".format(tgt_range, ref_range))

        # create attn mask of first layer of unet
        if mask_dims == 4:
            p2p_mask = torch.zeros((1, 1, tgt_size * 80, ref_size))
            p2p_mask[:, :, tgt_range[0] * 80:tgt_range[1] * 80, ref_range[0]:ref_range[1]] = 1
        else:
            raise IOError("NOT SURE!")

        # Create attn mask of other layers of unet
        num_resolutions = len(self.in_out)
        p2p_masks_dict["down"].append(p2p_mask)
        identity = torch.nn.Identity()
        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (num_resolutions - 1)
            p2p_mask = torch.nn.functional.interpolate(p2p_mask, scale_factor=0.5, mode="nearest") \
                if not is_last else identity(p2p_mask)
            if not is_last:
                p2p_masks_dict["down"].append(p2p_mask)
        p2p_masks_dict["mid"].append(p2p_mask)

        for ind, (dim_in, dim_out) in enumerate(reversed(self.in_out[1:])):
            p2p_mask = torch.nn.functional.interpolate(p2p_mask, scale_factor=2, mode="nearest")
            p2p_masks_dict["up"].append(p2p_mask)
        return p2p_masks_dict

    def create_p2p_mask2(self,
                         tgt_size: int = None,
                         ref_size: int = None,
                         mask_dims=4,
                         tgt_range=(10, 20),
                         ref_range=(10, 20)
                         ):
        """


        """
        p2p_masks_dict = {"down": [], "mid": [], "up": []}

        if tgt_range[1] > tgt_size or ref_range[1] > ref_size:
            print("Given tgt_range: {} or ref_range: {} is out of size".format(tgt_range, ref_range))

        # create attn mask of first layer of unet
        if mask_dims == 4:
            p2p_mask = torch.zeros((1, 1, tgt_size * 80, ref_size))
            p2p_mask[:, :, tgt_range[0] * 80:tgt_range[1] * 80, ref_range[0]:ref_range[1]] = 1
        else:
            raise IOError("NOT SURE!")

        # Create attn mask of other layers of unet
        num_resolutions = len(self.in_out)
        p2p_masks_dict["down"].append(p2p_mask)
        identity = torch.nn.Identity()
        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (num_resolutions - 1)
            p2p_mask = torch.nn.functional.interpolate(p2p_mask, scale_factor=0.5, mode="nearest") \
                if not is_last else identity(p2p_mask)
            if not is_last:
                p2p_masks_dict["down"].append(p2p_mask)
        p2p_masks_dict["mid"].append(p2p_mask)

        for ind, (dim_in, dim_out) in enumerate(reversed(self.in_out[1:])):
            p2p_mask = torch.nn.functional.interpolate(p2p_mask, scale_factor=2, mode="nearest")
            p2p_masks_dict["up"].append(p2p_mask)
        return p2p_masks_dict


    def create_p2p_mask(self,
                        tgt_size: int=None,
                        ref_size: int=None,
                        mask_dims=4,
                        tgt_range=(10, 20),
                        ref_range=(10, 20)
                        ):
        """
        create phoneme2phoneme mask
        p2p_mask: ([b,1], [c,1], d_q * l_q, [l_k, 1])
        """
        # check range out of range
        if tgt_range[1] > tgt_size or ref_range[1] > ref_size:
            print("Given tgt_range: {} or ref_range: {} is out of size".format(tgt_range, ref_range))

        # first mask
        if mask_dims == 4:
            p2p_mask = torch.zeros((1, 1, tgt_size * 80, ref_size))
            p2p_mask[:, :, tgt_range[0] * 80:tgt_range[1] * 80, ref_range[0]:ref_range[1]] = 1
        else:
            raise IOError("NOT SURE!")

        # all masks
        p2p_masks_dict = {"down": [], "mid": [], "up": []}
        for dmu, ratios in self.down_mid_up_ration.items():
            for c_ratio, lqk_ratio in ratios:
                lq_len = int(tgt_size * 80 * lqk_ratio)
                lk_len = int(ref_size * lqk_ratio)
                p2p_mask_down = torch.zeros((1, 1, lq_len, lk_len))
                p2p_mask_down[0, 0, :, :] = p2p_mask[0, 0, 0::int(1 / lqk_ratio), 0::int(1 / lqk_ratio)]    # Downsample matrix
                p2p_masks_dict[dmu].append(p2p_mask_down)
        return p2p_masks_dict

    def create_left_righ_mask(self):
        pass


def create_p2p_mask(
        tgt_size: int=None,
        ref_size: int=None,
        dims=4,
        tgt_range=(10, 20),
        ref_range=(10, 20),
):
    """
    create phoneme2phoneme mask
    p2p_mask: ([b,1], [c,1], l_q, [l_k, 1])
    """
    # check range out of range
    if tgt_range[1] > tgt_size or ref_range[1] > ref_size:
        print("Given tgt_range: {} or ref_range: {} is out of size".format(tgt_range, ref_range))
    if dims == 4:
        #p2p_mask = torch.zeros((1, 1, tgt_size * 80, ref_size))
        #p2p_mask[:, :, tgt_range[0]*80:tgt_range[1]*80, ref_range[0]:ref_range[1]] = 1
        p2p_mask = torch.ones((1, 1, tgt_size, ref_size))
        p2p_mask[:, :, tgt_range[0]:tgt_range[1], ref_range[0]:ref_range[1]] = 0
    else:
        print("NOT SURE!")
    return p2p_mask


def create_left_right_mask(b, heads, d_q, l_q, device="cuda", maskType="bin"):
    if maskType == "bin":
        if device == "cuda":
            temp_mask_left = torch.ones((b, heads, d_q * l_q, 1)).cuda()
            temp_mask_right = torch.ones((b, heads, d_q * l_q, 1)).cuda()
        else:
            temp_mask_left = torch.ones((b, heads, d_q * l_q, 1))
            temp_mask_right = torch.ones((b, heads, d_q * l_q, 1))
        temp_mask_left[:, :, :d_q * int(l_q * 0.5), :] = 0
        temp_mask_left = temp_mask_left > 0
        temp_mask_right[:, :, d_q * int(l_q * 0.5):, :] = 0
        temp_mask_right = temp_mask_right > 0
        return temp_mask_left, temp_mask_right
    else:
        if device == "cuda":
            temp_mask_left = torch.ones((b, heads, l_q, 1)).cuda()
            temp_mask_right = torch.ones((b, heads, l_q, 1)).cuda()
        else:
            temp_mask_left = torch.ones((b, heads, l_q, 1))
            temp_mask_right = torch.ones((b, heads, l_q, 1))
        temp_mask_left[:, :, :int(l_q * 0.5), :] = 0
        temp_mask_left = temp_mask_left > 0
        temp_mask_right[:, :, int(l_q * 0.5):, :] = 0
        temp_mask_right = temp_mask_right > 0
        return temp_mask_left, temp_mask_right


def create_utter_word_mask_freq(b, heads, d_q, l_q, emb_n,
                                ref_start, ref_end, target_start, target_end):
    """
    Create masks for utterance-level and word-level style transfering.
    """
    # create utter mask
    temp_mask_utter = torch.ones((b, heads, d_q * l_q, emb_n)).cuda()
    temp_mask_utter[:, :, target_start:target_end, :] = 0

    # create word mask
    temp_mask_word = torch.zeros((b, heads, d_q * l_q, emb_n)).cuda()
    temp_mask_word[:, :, target_start:target_end, ref_start:ref_end] = 1
    return temp_mask_utter, temp_mask_word


def create_low_high_mask_freq(b, heads, d_q, l_q,
                              freq_splitter="pitch"):
    """
    low high frequency mask
    """

    if freq_splitter == "pitch":
        split_mel_bin = change_freq_to_melbins(pitch_range[1])
    elif freq_splitter == "formant12":
        split_mel_bin = change_freq_to_melbins(f12_range[1])
    else:
        split_mel_bin = 0
        IOError("freq_splitter should be pitch or formant12")

    mask_low_freq = torch.zeros((b, heads, d_q * l_q, 1)).cuda()
    for i in l_q:
        mask_low_freq[:, :, i * d_q:i * ( d_q + split_mel_bin), :] = 1
    mask_high_freq = 1 - mask_low_freq
    return mask_low_freq, mask_high_freq


# https://iopscience.iop.org/article/10.1088/1742-6596/1153/1/012008/pdf
# freq_splitter = "pitch"
pitich_range_male = (100, 150)
pitch_range_female = (200, 250)
pitch_range = (90, 255)
non_pitch_range = (255, 5500)

# freq_splitter = "formant12"
f1_range = (650, 950)
f2_range = (1300, 1500)
f12_range = (650, 1500)

f3_range = (2500, 4000)
f4_range = (4500, 5500)  # not for sure
f34_range = (1500, 5500)

def change_freq_to_melbins(search_freq):
    freq_bins = librosa.mel_frequencies(n_mels=80)
    return bisect(freq_bins, search_freq)



# ------------------------- NOT USED-----------------------------
def create_left_right_mask_temp(b, heads, d_q, l_q):
    """
    left_mask_list = []
    right_mask_list = []
    for s in range(int(l_q * 0.5)):
        r_n = torch.arange(s, (l_q - 1) * d_q + s, d_q)
        left_mask_list.append(r_n)
    for s in range(l_q - int(l_q * 0.5)):
        r_n = torch.arange(l_q - s - 1, (l_q - 1) * d_q + l_q - s - 1, d_q)
        right_mask_list.append(r_n)

    left_mask_torch = torch.sort(torch.concat(left_mask_list, dim=1)).unsqueeze(-1)
    right_mask_list = torch.sort(torch.concat(right_mask_list, dim=1)).unsqueeze(-1)
    """

    temp_mask_left = torch.ones((b, heads, d_q * l_q, 1)).cuda()
    temp_mask_right = torch.ones((b, heads, d_q * l_q, 1)).cuda()
    temp_mask_left[:, :, :d_q * int(l_q * 0.5), :] = 0
    temp_mask_left = temp_mask_left > 0
    temp_mask_right[:, :, d_q * int(l_q * 0.5):, :] = 0
    temp_mask_right = temp_mask_right > 0
    return temp_mask_left, temp_mask_right

def create_pitch_bin_mask(wav_f,
                          mel_spectrogram,
                          n_mels=80,
                          fmin=0.0,
                          fmax=8000.0
                          ):
    """
    create pitch_bin mask given psd contours (Only in Inference)
    # change pitch_contour to pitch_bin_contour (denormalization to freq to bin)
    # pitch_bin_contour to pitch_bin_mask

    Args:
        wav_f:
        mel_spectrogram: (b, ?, 768)
        psd_contours: (b, 3, r)
        psd_mask : (b, 3)

    Returns:
        pitch_bin_mask = (b, c, d, l)

    """
    # Trim speech ?? no Need trim ?? -> if trim shorter than mel, no trim far greater than mel
    tg_dir = "/home/rosen/Project/FastSpeech2/preprocessed_data/ESD/TextGrid"
    wav_base = os.path.basename(wav_f).split(".")[0]
    tg_sub_dir = tg_dir + "/" + wav_base.split("_")[0]
    tg_f = tg_sub_dir + "/" + wav_base + ".TextGrid"

    pitch_contour = extract_pitch(wav_f, tg_f, need_trim=True)
    # TEMP: pad/remove pitch length to match with mel
    if len(mel_spectrogram.shape) == 3:
        mel_spectrogram = mel_spectrogram.squeeze()
    if len(pitch_contour) < mel_spectrogram.shape[0]:
        pitch_contour = np.append(pitch_contour, [0.0] * (len(mel_spectrogram) - len(pitch_contour)))
    elif len(pitch_contour) > mel_spectrogram.shape[0]:
        pitch_contour = pitch_contour[:mel_spectrogram.shape[0]]
    else:
        print("Pitch length match mel length")

    # change pitch_contour to pitch_bin_contour (denormalization -> ?)
    # pitch freq to mel bin given frequencies tuned to mel scale.
    freq_list = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax, htk=False)
    # Convert pitch hz to mel bin for each mel frame
    pitch_mel_bin_list = []  # (mel_n, )
    for pitch in pitch_contour:
        pitch_mel_bin = np.argmin(abs(freq_list - pitch))
        pitch_mel_bin_list.append(pitch_mel_bin)

    pitch_bin_mask = torch.zeros_like(mel_spectrogram)
    for fr_th in range(mel_spectrogram.shape[0]):
        # map the i-th frame to j-th pitch <- if len(mel) != len(pitch)
        #pit_th = pitmel_align_score[fr_th]
        fr_mel_bin = pitch_mel_bin_list[fr_th]
        if fr_mel_bin != 0:  # Only for pitch existed!
            pitch_bin_mask[fr_th, fr_mel_bin] = 1
    return pitch_bin_mask


def create_left_right_mask_score(b, heads, d_q, l_q, device="cuda"):
    if device == "cuda":
        temp_mask_left = torch.zeros((b, heads, d_q * l_q, 1)).cuda()
        temp_mask_right = torch.zeros((b, heads, d_q * l_q, 1)).cuda()
    else:
        temp_mask_left = torch.zeros((b, heads, d_q * l_q, 1))
        temp_mask_right = torch.zeros((b, heads, d_q * l_q, 1))

    temp_left_score = torch.linspace(0, 1, steps=d_q * int(l_q * 0.5)).unsqueeze(0).transpose(0, 1)
    temp_right_score = torch.linspace(0, 1, steps=d_q * l_q - d_q * int(l_q * 0.5)).unsqueeze(0).transpose(0, 1)

    temp_left_score = temp_left_score.repeat(b, heads, 1, 1)
    temp_right_score = temp_right_score.repeat(b, heads, 1, 1)

    temp_mask_left[:, :, :d_q * int(l_q * 0.5), :] = temp_left_score
    temp_mask_right[:, :, d_q * int(l_q * 0.5):, :] = temp_right_score
    return temp_mask_left, temp_mask_right
