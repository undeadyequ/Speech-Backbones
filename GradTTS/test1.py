import sys
import torch
from einops import rearrange

from scipy.io import wavfile
import pyworld as pw
import numpy as np
import matplotlib.pyplot as plt

import tgt

def test_qkv():
    q = torch.randn((2, 4, 2, 3, 4))
    k = torch.randn((2, 4, 2, 4, 4))

    #attn = torch.bmm(q, k)


    context = torch.einsum('bhkdn,bhken->bhkde', q, k)

    qkv = torch.randn((2, 18, 4, 4))
    q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)',
                        heads=2, qkv=3)
    print(q.shape, k.shape, v.shape)


def get_alignment(tier, sampling_rate=16000, hop_length=256):
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


test_wav_f = "/home/rosen/Project/FastSpeech2/ESD/16k_wav/0011_000151.wav"
test_wav_mel = "/home/rosen/Project/FastSpeech2/preprocessed_data/ESD/mel/0011-mel-0011_000151.npy"
test_wav_pitch = "/home/rosen/Project/FastSpeech2/preprocessed_data/ESD/pitch/0011-pitch-0011_000151.npy"
test_wav_tg_path = "/home/rosen/Project/FastSpeech2/preprocessed_data/ESD/TextGrid/0011/0011_000151.TextGrid"
# fs : sampling frequency, 音楽業界では44,100Hz
# data : arrayの音声データが入る
import librosa

def test_pitch_extraction():
    # Get alignments
    textgrid = tgt.io.read_textgrid(test_wav_tg_path)
    phone, duration, start, end = get_alignment(
        textgrid.get_tier_by_name("phones")
    )

    print(duration)

    wav, fs = librosa.load(test_wav_f)
    wav = wav[
          int(fs * start): int(fs * end)
          ].astype(np.float64)


    # floatでないとworldは扱えない
    _f0, _time = pw.dio(wav, fs,
                        frame_period=256 / 16000 * 1000,
                        )    # 基本周波数の抽出
    f0 = pw.stonemask(wav, _f0, _time, fs)  # 基本周波数の修正

    print(fs, len(wav))
    print(len(_f0))
    print("len of time", len(_time))
    print("len of pitch:", len(f0))

    # mel
    mel = np.load(test_wav_mel)
    print(mel.shape)
    p = np.load(test_wav_pitch)
    print(p.shape)

    # --- これは音声の合成に用いる(今回は使わない)
    # sp = pw.cheaptrick(data, f0, _time, fs)  # スペクトル包絡の抽出
    # ap = pw.d4c(data, f0, _time, fs)         # 非周期性指標の抽出
    # y = pw.synthesize(f0, sp, ap, fs)    # 合成


    # 可視化
    #plt.plot(data, label="Raw Data")
    #plt.legend(fontsize=10)
    #plt.show();


    plt.plot(f0, linewidth=3, color="green", label="F0 contour")
    plt.legend(fontsize=10)
    plt.show();

    """
    plt.plot(_f0, linewidth=3, color="blue", label="F0 contour")
    plt.legend(fontsize=10)
    plt.show();
"""




