import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text
import json

def prepare_align(config):
    """
    prepare .wav and .lab pairs for montreal aligment
    """
    in_dir = config["path"]["corpus_path"]
    in_wav_dir = config["path"]["wav_dir"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    meta_data = os.path.join(in_dir, "metadata.json")
    with open(meta_data, 'r') as f:
        meta_dict = json.load(f)
        for id, data_dict in meta_dict.items():
            # get each value
            speaker = os.path.basename(id).split("_")[0]
            base_name = os.path.basename(id)
            text = data_dict["text"]
            text = _clean_text(text, cleaners)
            emo = data_dict["emotion"]

            wav_path = os.path.join(in_dir, in_wav_dir, "{}.wav".format(base_name))
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)
            else:
                print("not exist {}".format(wav_path))