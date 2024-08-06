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


if __name__ == '__main__':
    # test MCD

    # instance of MCD class
    # three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")

    # two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively
    mcd_value = mcd_toolbox.calculate_mcd("../exp/sample_21st.wav", "../exp/sample_22nd.wav")
    print(mcd_value)

