# if run under test folder
import sys
import librosa
import matplotlib.pyplot as plt


sys.path.append('/Users/luoxuan/Project/Speech-Backbones/GradTTS')
import torch
import yaml
from GradTTS.text.symbols import symbols

from GradTTS.model.estimators import GradLogPEstimator2dCond, \
    create_pitch_bin_mask, create_left_right_mask


def test_create_pitch_bin_mask(

):
    wav_f = ""
    y, sr = librosa.load(wav_f)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_mels=128,
                                                     fmax=8000)
    pitch_bin_mask = create_pitch_bin_mask(wav_f,
                                           mel_spectrogram,
                                           n_mels=80,
                                           fmin=0.0,
                                           fmax=8000.0,
                                           need_trim=True)

    # draw melspectrogram

    # draw mask
    fig, ax = plt.subplots()

    img = librosa.display.specshow(pitch_bin_mask,
                                   x_axis='time',
                                   y_axis='mel', sr=sr,
                                   fmax=8000,
                                   ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='pitch bin mask')
    fig.savefig("pitch_bin_mask.png")


if __name__ == '__main__':
    test_create_pitch_bin_mask()