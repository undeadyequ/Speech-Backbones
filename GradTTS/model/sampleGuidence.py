import os.path

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from GradTTS.model.base import BaseModule

import math
from einops import rearrange, repeat

from GradTTS.model.estimators_v2 import GradLogPEstimator2dCond
import librosa
from GradTTS.model.utils import extract_pitch

class SampleGuidence(BaseModule):
    """
    Guide sample in order to guide attention ma
    """
    def __init__(self,
                 dim,
                 sample_channel_n,
                 n_spks,
                 spk_emb_dim,
                 pe_scale,
                 emo_emb_dim,
                 att_type,
                 att_dim,
                 heads,
                 p_uncond
                 ):
        super(SampleGuidence, self).__init__()
        self.estimator = GradLogPEstimator2dCond(dim,
                                                 sample_channel_n=sample_channel_n,
                                                 n_spks=n_spks,
                                                 spk_emb_dim=spk_emb_dim,
                                                 pe_scale=pe_scale,
                                                 emo_emb_dim=emo_emb_dim,
                                                 att_type=att_type,
                                                 att_dim=att_dim,
                                                 heads=heads,
                                                 p_uncond=p_uncond,
                                                 )

    def guide_sample_interp(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            mu: torch.Tensor,
            t,
            spk,
            melstyle1,
            emo_label1,
            pitch1,
            melstyle2,
            emo_label2,
            pitch2,
            align_len,
            align_mtx,
            guide_scale: float,
            tal=40,
            alpha=0.08
    ):
        """
        Guide sample with two ref audios (One ref for pitch, other for NONPitch part)
        Args:
            x:
            guid_mask:
            mask:
            mu:
            t:
            spk:
            melstyle1:
            emo_label1:
            melstyle2:
            emo_label2:
            guide_scale:
            tal:
            alpha:

        Returns:

        """
        if t < tal:
            guid_mask = create_pitch_bin_mask(
                pitch1,
                melstyle1
            )  # get guid mask from melstyle or prosodic contours ?
            x_score = self.guide_sample(
                x,
                guid_mask,
                mask,
                mu,
                t,
                spk,
                melstyle1,
                emo_label1,
                align_len,
                align_mtx,
                guide_scale,
                alpha
            )
        else:
            x_score = self.estimator(
                x, mask, mu, t, spk, melstyle2, emo_label2)
        return x_score

    def guide_sample(
            self,
            x: torch.Tensor,
            guid_mask: torch.Tensor,
            mask: torch.Tensor,
            mu: torch.Tensor,
            t,
            spk,
            melstyle,
            emo_label,
            align_len,
            align_mtx,
            guide_scale: float,
            tal=40,
            alpha=0.08
    ):
        """
        Guide sample by single ref audio

        # set x to require_grad
        # set optimizer with x
        # set loss = attn_in_mask + attn_out_mask
        # get x_guided
        # get x_guided_score = unet(x_guided)

        Args:
            x ():
            mask ():
            mu ():
            cond ():

        Returns:
            x_guided_score
        """
        loss = 0
        # set x to require_grad
        x_guided = x.clone().detach()
        x_guided.requires_grad = True

        # set optimizer with x
        optim = torch.optim.SGD([x_guided], lr=alpha)
        # set loss = attn_in_mask + attn_out_mask
        _, attn_score = self.estimator(x_guided,
                                       mask,
                                       mu,
                                       t,
                                       spk,
                                       psd=None,
                                       melstyle=melstyle,
                                       emo_label=emo_label,
                                       align_len=align_len,
                                       align_mtx=align_mtx
                                       )
        loss += -attn_score[guid_mask == 1].sum()
        loss += guide_scale * attn_score[guid_mask == 0].sum()
        loss.backward()

        # get x_guided
        optim.step()

        # get x_guided_score = unet(x_guided)
        x_guided_score = self.estimator(x_guided, mask, mu, t, spk, melstyle, emo_label)
        return x_guided_score, loss


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


    # pitch_mel_bin_list to pitch_bin_mask ??
    ## align pitch (word level?) contours with mel spectrogram (frame-level)?
    #pitmel_align_score = align_pitch_mel(pitch_contour, mel_spectrogram.squeeze(0))

    ## create pitchMask:
    #mel_spectrogram = torch.from_numpy(mel_spectrogram)
    #pitch_contour = torch.from_numpy(pitch_contour)

    pitch_bin_mask = torch.zeros_like(mel_spectrogram)
    for fr_th in range(mel_spectrogram.shape[0]):
        # map the i-th frame to j-th pitch <- if len(mel) != len(pitch)
        #pit_th = pitmel_align_score[fr_th]
        fr_mel_bin = pitch_mel_bin_list[fr_th]
        if fr_mel_bin != 0:  # Only for pitch existed!
            pitch_bin_mask[fr_th, fr_mel_bin] = 1
    return pitch_bin_mask



def align_pitch_mel(
        pitch,
        mel
):
    """

    Args:
        pitch(M,):
        t(M,): the time of each pitch point
        mel(N,):
    Returns:
        align_score(N):

        Exp: N = 5, M = 3
        align_score = [0, 0, 0, 1, 2]
    """
    align_score = np.zeros_like(mel[:, 0], dtype=np.int16)

    # for test
    """
    align_score[-3] = 1
    align_score[-2] = 2
    align_score[-1] = 3
    """
    return torch.from_numpy(align_score)


if __name__ == '__main__':
    # test
    create_pitch_bin_mask(

    )
    # SampleGuidence = SampleGuidence()

