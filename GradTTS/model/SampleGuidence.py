import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import math
from einops import rearrange, repeat

from GradTTS.model.estimators import GradLogPEstimator2dCond
import librosa


class SampleGuidence:
    """
    Guide sample in order to guide attention map
    """
    def __init__(self,
                 dim,
                 sample_channel_n,
                 n_spks,
                 spk_emb_dim,
                 pe_scale,
                 emo_emb_dim,
                 att_type
                 ):
        self.estimator = GradLogPEstimator2dCond(dim,
                                                 sample_channel_n=sample_channel_n,
                                                 n_spks=n_spks,
                                                 spk_emb_dim=spk_emb_dim,
                                                 pe_scale=pe_scale,
                                                 emo_emb_dim=emo_emb_dim,
                                                 att_type=att_type,
                                                 )
    def guide_sample(
            self,
            x: torch.Tensor,
            guid_mask: torch.Tensor,
            mask: torch.Tensor,
            t,
            mu: torch.Tensor,
            spk,
            melstyle,
            emo_label,
            guide_scale: float,
            tal=40,
            alpha=0.08
    ):
        """
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
        if t < tal:
            # set x to require_grad
            x_guided = x.clone().detach()
            x_guided.require_grad()

            # set optimizer with x
            optim = torch.optim.SGD([x_guided], lr=alpha)
            # set loss = attn_in_mask + attn_out_mask
            _, attn_score = self.estimator(x_guided, mask, mu, t, spk, melstyle, emo_label)
            loss += -attn_score[guid_mask == 1].sum()
            loss += guide_scale * attn_score[guid_mask == 0].sum()
            loss.backward()

            # get x_guided
            optim.step()

            # get x_guided_score = unet(x_guided)
            x_guided_score = self.estimator(x_guided, mask, mu, t, spk, melstyle, emo_label)
        else:
            x_guided_score = self.estimator(x, mask, mu, t, spk, melstyle, emo_label)
        return x_guided_score, loss


def create_pitch_bin_mask(psd_contours,
                          psd_mask,
                          n_mels=80,
                          fmin=0.0,
                          fmax=8000.0
                          ):
    """
    create pitch_bin mask given psd contours (Only in Inference)
    # change pitch_contour to pitch_bin_contour (denormalization to freq to bin)
    # pitch_bin_contour to pitch_bin_mask

    Args:
        psd_contours: (b, 3, r)
        psd_mask : (b, 3)

    Returns:
        pitch_bin_mask = (b, c, d, l)

    """
    # change pitch_contour to pitch_bin_contour (denormalization -> ?)
    ## de-normalization to freq ?
    pitch_contour = psd_contours[:, 1, :][psd_mask[:, 1]]
    pitch_mean = 1

    ## freq to bin ?? can mel_bin directly compute from frequency ?
    freq_list = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax, htk=False)

    # pitch_bin_contour to pitch_bin_mask ??
    pitch_bin_mask = torch.ones([])

    return pitch_bin_mask




if __name__ == '__main__':
    SampleGuidence = SampleGuidence()
