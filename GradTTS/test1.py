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

import subprocess
def test_mfa():
    in_wavtxt_dir = "~/data/test_data/"
    out_textgrid_dir = "~/data/test_data/"
    model = "english_mfa"
    venv_python = '/home/rosen/anaconda3/envs/aligner/bin/python'
    args = [venv_python, 'mfa_dir.py', in_wavtxt_dir, model, out_textgrid_dir]
    subprocess.run(args)



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


if __name__ == '__main__':
    test_mfa()

