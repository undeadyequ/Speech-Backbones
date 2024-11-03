from pymcd.mcd import Calculate_MCD
import librosa
import soundfile as sf


# instance of MCD class
# three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics

mcd_toolbox = Calculate_MCD(MCD_mode="MCD-DTW")

# two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively
#mcd_value = mcd_toolbox.calculate_mcd("001.wav", "002.wav")


import librosa
from pymcd.mcd import Calculate_MCD
import torch
import tgt
import numpy as np
import os
from GradTTS.const_param import textgrid_dir

def get_pitch_match_score(pitch1, pitch2):
    pitch_score = 0
    return pitch_score


def get_pitch_match_score(pitch1, pitch2):
    pitch_score = 0
    return pitch_score


def get_ref_pRange(p_start=5, p_end=8, ref_id="0019_000403"):
    """
    # S(5) T(6) IH1(7) L(8)
    get range of specific phoneme on frames by given phoneme index of reference speech id
    """
    # get the time/frame range of given ref text (Know from ref id)
    speaker = ref_id.split("_")[0]
    tg_path = os.path.join(
        textgrid_dir, speaker, "{}.TextGrid".format(ref_id)
    )
    textgrid = tgt.io.read_textgrid(tg_path)
    # duration = frame number
    phone, duration, start, end = get_alignment(
        textgrid.get_tier_by_name("phones")
    )
    #pIndex = get_pIndex_from_wIndx(ref_wIndx) # ?? word ??
    ref_nRange = (sum(duration[:p_start]), sum(duration[:p_end+1]))
    #frame_n = sum(duration)
    # check if ref_nRange match the spefied frame (still) ??
    return ref_nRange


def get_tgt_pRange():
    pass

def get_pIndex_from_wIndx(ref_wIndx):
    pass


def get_alignment(tier):
    sil_phones = ["sil", "sp", "spn"]

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    sampling_rate = 16000
    hop_length = 256

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

def compute_phoneme_mcd(wav1,
                        wav2,
                        p1_interv=(0, 1),
                        p2_interv=(0, 1),
                        ):

    # read wav1, wav2
    y1, sr1 = librosa.load(wav1, sr=None)
    y2, sr2 = librosa.load(wav2, sr=None)
    # split
    y1_split = y1[p1_interv[0] * sr1 : p1_interv[1]]
    y2_split = y2[p2_interv[0] * sr2 : p2_interv[1]]

    # write to file
    wav1_split = wav1.split(".")[0] + "_p{}_{}.wav".format(
        p1_interv[0], p1_interv[1])
    wav2_split = wav2.split(".")[0] + "_p{}_{}.wav".format(
        p2_interv[0], p2_interv[1])
    sf.write(wav1_split, y1_split, sr1)
    sf.write(wav2_split, y2_split, sr2)
    # mcd
    mcd_toolbox.calculate_mcd(wav1_split, wav2_split)

def convert_frameNum2second(frame_num, sr=16000, chunk_len=8000):
    return frame_num * chunk_len / sr

if __name__ == '__main__':
    speech_dir = ("/home/rosen/Project/Speech-Backbones/GradTTS/logs/interpEmoTTS_frame2binAttn_noJoint/"
                  "chpt400_time50_spk19_emoAngry")
    non_p2p = speech_dir + "/no_p2p_sample_v1_0.wav"
    p2p = speech_dir + "/p2p_sample_v1_0.wav"

    origin_dir = "/home/rosen/Project/FastSpeech2/ESD/16k_wav"
    origin = origin_dir + "/0019_000403.wav"

    mcd_origin_nonP2P = compute_phoneme_mcd(
         non_p2p,
        origin,
        p1_interv=(0, 1),
        p2_interv=(0, 1)
    )

    mcd_origin_p2p = compute_phoneme_mcd(
        p2p,
        origin,
        p1_interv=(0, 1),
        p2_interv=(9, 1)
    )

    print("mcd betwenn origin and nonp2p is: {}".format(mcd_origin_nonP2P))
    print("mcd betwenn origin and p2p is: {}".format(mcd_origin_p2p))


