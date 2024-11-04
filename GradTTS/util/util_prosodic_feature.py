import math
from typing import Any, Dict, Optional, List, Tuple

import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from GradTTS.util.util_silenceremove import remove_silence_from_wav


def extract_pitch(
        audio_f: Optional[str] = None,
        audio_list: Optional[List[Any]] = None,
        fs: Optional[int] = None,
        otype: str = "f0"
):
    """

    Args:
        audio_f:
        audio_list:
        fs:
        otype:

    Returns:

    """
    pass

def extract_energy():
    """

    Returns:

    """
    pass


def extract_emo_feature_modify(
        audio: str,
        sr: int = 22050,
        remove_silence: bool = False,
        #min_max_stats_f: str = "/home/rosen/Project/espnet/test_utils/test_csv/blizzard13/train_no_dev/id_pth_fts_norm_stats.csv",
        min_max_stats_f=None,
        short_version=True,
        return_contour=False
):
    """
    Extract emotional features showed in paper
    Multimodal Speech Emotion Recognition and Ambiguity Resolution
    https://arxiv.org/pdf/1904.06022.pdf

    rmse:
    pitch:
    harmonic:
    pitch:

    normalize: if normalize needed

    audio: audio file or audio list
    return feature_list: np of [n_samples, n_features]
    """
    #st = time.time()

    feature_list = []
    if isinstance(audio, str):
        if remove_silence:
            temp_f = "temp.wav"
            remove_silence_from_wav(audio, agress=1, out_wav=temp_f)
            y, _ = librosa.load(temp_f, sr)
            #os.remove(temp_f)
        else:
            y, _ = librosa.load(audio, sr)
    else:
        y = audio
    #print("read time:{}".format(st1-st))

    # 2. rmse
    rmse = librosa.feature.rms(y + 0.0001)[0]
    lg_rmse = [20 * math.log(r) / math.log(10) for r in rmse]
    feature_list.append(np.mean(lg_rmse))  # rmse_mean
    feature_list.append(np.std(lg_rmse))  # rmse_std
    feature_list.append(np.max(lg_rmse) - np.min(lg_rmse))  # rmse_range
    # st2 = time.time()
    #print("rmse time:{}".format(st2-st1))
    # rmse(mean, std, max), harmonic(mean, std), pitch(mean, std, max)

    # 3. harmonic
    if not short_version:
        y_harmonic = librosa.effects.hpss(y)[0]
        feature_list.append(np.mean(y_harmonic if y_harmonic is not np.NAN else 0) * 1000 )  # harmonic (scaled by 1000)
        feature_list.append(np.std(y_harmonic if y_harmonic is not np.NAN else 0) * 1000 )  # harmonic (scaled by 1000)

    # 4. pitch (instead of auto_correlation)
    pitch, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    ZERO_ALLOW = True
    if ZERO_ALLOW:
        pitch = [0 if math.isnan(p) else p for p in pitch]
        lg_pitch_nzero = [math.log(p) / math.log(10) if p != 0 else 0 for p in pitch]
    else:
        lg_pitch_nzero = [math.log(p) / math.log(10) for p in pitch if not math.isnan(p) and p != 0]
    if len(lg_pitch_nzero) == 0:
        lg_pitch_nzero = [0]
    feature_list.append(np.mean(lg_pitch_nzero))
    feature_list.append(np.std(lg_pitch_nzero))
    feature_list.append(np.max(lg_pitch_nzero) - np.min(lg_pitch_nzero))
    feature_list = np.round(np.array(feature_list).reshape(1, -1), 3)

    #st4 = time.time()
    #print("pitch time:{}".format(st4-st3))

    # Check Normalization
    if min_max_stats_f is not None:
        scalar = MinMaxScaler()
        train_feats_stats = pd.read_csv(min_max_stats_f)
        scalar.fit(train_feats_stats)
        feature_list = scalar.transform(feature_list)

    if return_contour:
        return lg_rmse, lg_pitch_nzero, feature_list
    else:
        return feature_list


