import os
import random
import json
from os import times
from os.path import basename

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import GradTTS.audio as Audio
from typing import Any, Dict, Optional, List, Tuple
import time


class PreprocessorExtract:
    """
    rewrite the Preprocessor class
    """
    def __init__(self, config):
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def average_phoneme(self, pitch, duration):
        # perform linear interpolation
        nonzero_ids = np.where(pitch != 0)[0]
        interp_fn = interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
            bounds_error=False,
        )
        pitch = interp_fn(np.arange(0, len(pitch)))

        # Phoneme-level average
        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                pitch[i] = np.mean(pitch[pos: pos + d])
            else:
                pitch[i] = 0
            pos += d
        pitch = pitch[: len(duration)]
        return pitch

    def get_alignment(self, tier):
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
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def extract_pitch_energy_mel(self, wav_path, tg_path=None, out_dir=None, average_phoneme=True, save_npy=True):
        """

        Args:
            wav_path:
            average_phoneme:
            tg_dir:
        Returns:

        """

        # setup output dir
        out_dir_pitch = out_dir + "/pitch"
        out_dir_energy = out_dir + "/energy"
        out_dir_mel = out_dir + "/mel"

        os.makedirs((out_dir_pitch), exist_ok=True)
        os.makedirs((out_dir_energy), exist_ok=True)
        os.makedirs((out_dir_mel), exist_ok=True)

        basename = os.path.basename(wav_path).split(".")[0]

        # read tg files

        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        phonemes = " ".join(phone)
        if start >= end:
            raise IOError("Wrong textgrid file")
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path, sr=None)
        wav = wav.astype(np.float32)
        wav = wav[
              int(self.sampling_rate * start): int(self.sampling_rate * end)
              ].astype(np.float32)

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            raise IOError("Wrong pitch extracted file!")
            return None

        # Compute mel and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)

        # average phoneme
        if average_phoneme:
            pitch = self.average_phoneme(pitch, duration)
            energy = self.average_phoneme(energy, duration)

        # Save files
        #print("pitch size:", pitch.size)
        if save_npy:
            pitch_filename = "{}.npy".format( basename)
            energy_filename = "{}.npy".format(basename)
            mel_filename = "{}.npy".format(basename)

            np.save(os.path.join(out_dir_pitch, pitch_filename), pitch)
            np.save(os.path.join(out_dir_energy, energy_filename), energy)
            np.save(
                os.path.join(out_dir_mel, mel_filename),
                mel_spectrogram.T)

        #pitch_rmn = self.remove_outlier(pitch)
        #energy_rmn = self.remove_outlier(energy)

        #print("pitch len: {}/{}, dur len: {}".format(len(pitch), len(pitch_rmn), len(duration)))

        return (
            #"|".join([basename, phonemes]),
            phonemes,
            pitch,
            energy,
            mel_spectrogram,
            duration
        )

    def extract_energy(self, wav_path):
        """
        basename: 0011_000002

        Args:
            wav_path:

        Returns:

        """
        # output
        out_dir_energy = os.path.dirname(wav_path) + "_out/energy"
        out_dir_mel = os.path.dirname(wav_path) + "_out/mel"

        os.makedirs((out_dir_energy), exist_ok=True)
        os.makedirs((out_dir_mel), exist_ok=True)

        basename = os.path.basename(wav_path).split(".")[0]
        #speaker = basename.split("_")[0]

        wav, _ = librosa.load(wav_path, sr=None)
        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)

        energy_filename = "{}.npy".format( basename)
        np.save(os.path.join(out_dir_energy, energy_filename), energy)

        mel_filename = "{}.npy".format(basename)
        np.save(
            os.path.join(out_dir_mel, mel_filename),
            mel_spectrogram.T
        )

        print("energy size:", energy.size)

        return (
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)
        return values[normal_indices]

