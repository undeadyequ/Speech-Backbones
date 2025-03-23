# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
import json
import random
import numpy as np

import torch
import torchaudio as ta

from text import text_to_sequence, cmudict, phoneme_to_sequence
from text.symbols import symbols
from utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility
from params import seed as random_seed

import sys
sys.path.insert(0, 'hifi-gan')
from meldataset import mel_spectrogram
import os

emo_num = {
    "Angry": 0,
    "Surprise": 1,
    "Sad": 2,
    "Neutral": 3,
    "Happy": 4
}

class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        random.seed(random_seed)
        random.shuffle(self.filepaths_and_text)

    def get_pair(self, filepath_and_text):
        filepath, text = filepath_and_text[0], filepath_and_text[1]
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        return (text, mel)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        text, mel = self.get_pair(self.filepaths_and_text[index])
        item = {'y': mel, 'x': text}
        return item

    def __len__(self):
        return len(self.filepaths_and_text)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_ = item['y'], item['x']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths}


class TextMelSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.add_blank = add_blank
        random.seed(random_seed)
        random.shuffle(self.filelist)

    def get_triplet(self, line):
        filepath, text, speaker = line[0], line[1], line[2]
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        speaker = self.get_speaker(speaker)
        return (text, mel, speaker)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_speaker(self, speaker):
        speaker = torch.LongTensor([int(speaker)])
        return speaker

    def __getitem__(self, index):
        text, mel, speaker = self.get_triplet(self.filelist[index])
        item = {'y': mel, 'x': text, 'spk': speaker}
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        """

        Args:
            size:

        Returns:
            test_batch: list(dict("spk":"", ), ...)
        """
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelSpeakerEmoDataset(torch.utils.data.Dataset):
    """
    Read Text mel, spker, and emotion
    """
    def __init__(self, filelist_path, meta_path, cmudict_path, preprocess_dir, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000,
                 datatype="psd", psd_gran="frame", need_rm_sil=True):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        with open(meta_path, "r") as f:
            self.metajson = json.load(f)
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.preprocessed_path = preprocess_dir
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.add_blank = add_blank
        self.datatype = datatype
        self.psd_gran = psd_gran
        self.need_rm_sil = need_rm_sil
        random.seed(random_seed)
        random.shuffle(self.filelist)

    def get_triplet(self, line):
        basename, speaker, phone, text = line[0], line[1], line[2], line[3]
        text = self.get_text(text)
        mel = self.get_mel(basename, read_exist=True, spk=speaker)
        spk = self.get_speaker(speaker)
        return (text, mel, spk)

    def get_mel(self, filepath, read_exist=True, spk=None):
        if read_exist:
            mel_path = os.path.join(
                self.preprocessed_path,
                "mel",
                "{}-mel-{}.npy".format(spk, filepath),
            )
            mel = torch.from_numpy(np.load(mel_path))
            mel = torch.transpose(mel, 0, 1)
        else:
            audio, sr = ta.load(filepath)
            assert sr == self.sample_rate
            mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                                  self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_emo(self, basename):
        iiv_path = os.path.join(
            self.preprocessed_path,
            #"iiv_reps",
            "iiv_reps_iteronly",
            "{}.npy".format(basename),
        )
        iiv = torch.from_numpy(np.load(iiv_path).squeeze(0))
        return iiv

    def get_pit(self, basename, speaker):
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = torch.unsqueeze(torch.from_numpy(np.load(pitch_path)).to(torch.float32), 0)
        return pitch

    def get_eng(self, basename, speaker):
        eng_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        eng = torch.unsqueeze(torch.from_numpy(np.load(eng_path)).to(torch.float32), 0)
        return eng

    def get_dur(self, basename, speaker):
        dur_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        dur = torch.unsqueeze(torch.from_numpy(np.load(dur_path)).to(torch.float32), 0)
        return dur

    def get_wav2vect(self, basename, speaker):
        w2v_path = os.path.join(
            self.preprocessed_path,
            "emo_reps",
            "{}.npy".format(basename),
        )
        w2v = torch.unsqueeze(torch.from_numpy(np.load(w2v_path)), 0).transpose(1, 2)
        return w2v

    def get_facodec_psd(self, basename, speaker):
        facodec_f = os.path.join(
            self.preprocessed_path,
            "psd_quants",
            "{}.npy".format(basename),
        )
        facodec = torch.from_numpy(np.load(facodec_f))
        return facodec

    def get_text(self, text):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_phoneme(self, phoneme):
        text_norm = phoneme_to_sequence(phoneme)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_speaker(self, speaker):
        speaker = torch.LongTensor([int(speaker)])
        return speaker

    def __getitem__(self, index):
        basename, speaker, phone, text = self.filelist[index][0:4]
        text, mel, spk = self.get_triplet(self.filelist[index])
        item = {'y': mel, 'x': text, 'spk': spk}

        if self.datatype == "emo_embed":                   # iiv emb
            item["emo"] = self.get_emo(basename)
        elif self.datatype == "psd":                       # psd
            pit, eng, dur = (self.get_pit(basename, speaker),  # (1, p_l)
                             self.get_eng(basename, speaker),
                             self.get_dur(basename, speaker))
            if self.need_rm_sil:
                sil_idx = [i for i, p in enumerate(phone[1:-1].split()) if p != "sil" and p!= "spn"]
                pit, eng, dur = pit[:, sil_idx], eng[:, sil_idx], dur[:, sil_idx]
            if self.psd_gran == "frame":
                pit = torch.repeat_interleave(pit, repeats=dur.to(torch.long).squeeze(0), dim=1)
                eng = torch.repeat_interleave(eng, repeats=dur.to(torch.long).squeeze(0), dim=1)
                dur_frame = torch.arange(0, dur.shape[1]).unsqueeze(0)
                dur_frame = torch.repeat_interleave(dur_frame, repeats=dur.to(torch.long).squeeze(0), dim=1)  # 00...111
            emolabel = self.metajson[self.filelist[index][0]]["emotion"]
            emolabel = torch.tensor([emo_num[emolabel]], dtype=torch.long)
            melstyle = torch.stack([pit, eng, dur_frame], dim=1)
            item["dur"] = dur
            item["melstyle"] = melstyle
            item["emo_label"] = emolabel

        elif self.datatype == "melstyle":                   # wav2vect2
            melstyle = self.get_wav2vect(basename, speaker)
            emolabel = self.metajson[self.filelist[index][0]]["emotion"]
            emolabel = torch.tensor([emo_num[emolabel]], dtype=torch.long)
            item["melstyle"] = melstyle
            item["emo_label"] = emolabel

        elif self.datatype == "melstyle_onehot":            # wav2vect2 + emo_onehot
            melstyle = self.get_wav2vect(basename, speaker)
            emolabel = self.metajson[self.filelist[index][0]]["emotion"]
            emolabel = torch.nn.functional.one_hot(torch.tensor([emo_num[emolabel]]), num_classes=5)
            emolabel = torch.unsqueeze(emolabel, 0)
            item["melstyle"] = melstyle
            item["emo_label"] = emolabel

        elif self.datatype == "mel":                        # mel
            emolabel = self.metajson[self.filelist[index][0]]["emotion"]
            emolabel = torch.tensor([emo_num[emolabel]], dtype=torch.long)
            item["melstyle"] = mel.unsqueeze(0)
            item["emo_label"] = emolabel

        elif self.datatype == "FACodec":                        # FACodec
            melstyle = self.get_facodec_psd(basename, speaker)
            emolabel = self.metajson[self.filelist[index][0]]["emotion"]
            emolabel = torch.tensor([emo_num[emolabel]], dtype=torch.long)
            item["melstyle"] = melstyle   # 0 : (1, 256, L)
            item["emo_label"] = emolabel
        elif self.datatype == "FACodecDur":
            melstyle = self.get_facodec_psd(basename, speaker)
            emolabel = self.metajson[self.filelist[index][0]]["emotion"]
            emolabel = torch.tensor([emo_num[emolabel]], dtype=torch.long)
            dur = self.get_dur(basename, speaker)
            item["melstyle"] = melstyle   # 0 : (1, 256, L)
            item["emo_label"] = emolabel
            if self.need_rm_sil:
                sil_idx = [i for i, p in enumerate(phone[1:-1].split()) if p != "sil" and p!= "spn"]
                dur = dur[:, sil_idx]
            item["dur"] = dur
        else:
            item = {}
            print("Not implemented!")
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        """

        Args:
            size:

        Returns:
            test_batch: list(dict("spk":"", ), ...)
        """
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelSpeakerBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []
        spk = []

        for i, item in enumerate(batch):
            y_, x_, spk_ = item['y'], item['x'], item['spk']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            spk.append(spk_)

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk = torch.cat(spk, dim=0)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'spk': spk}


class TextMelSpeakerEmoBatchCollate(object):
    """
    Collate samples
    """
    def __call__(self, batch):
        B = len(batch)
        # x, y
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        # emo_label
        if "emo_label" in batch[0].keys():
            if len(batch[0]["emo_label"].shape) == 1:
                emo = torch.zeros((B,), dtype=torch.long)
            else:
                emo = torch.zeros((B, batch[0]["emo_label"].shape[-1]), dtype=torch.float32)


        if "emo" in batch[0].keys():
            emo_emb = torch.zeros((B, batch[0]["emo"].shape[-1]), dtype=torch.float32)        # Set length of sequential dataset
        ## psd
        if "pit" in batch[0].keys():
            pit_max_length = max([item["pit"].shape[-1] for item in batch])
            pits = torch.zeros((B, pit_max_length), dtype=torch.float32)
            engs = torch.zeros((B, pit_max_length), dtype=torch.float32)
            durs = torch.zeros((B, pit_max_length), dtype=torch.float32)
        if "melstyle" in batch[0].keys():
            melstyle_max_length = max([item["melstyle"].shape[2] for item in batch])
            melstyle_dim = batch[0]["melstyle"].shape[1]
            melstyles = torch.zeros((B, melstyle_dim, melstyle_max_length), dtype=torch.float32)

        y_lengths, x_lengths = [], []
        spk = []

        # Append batch list data to dictionary with batch tensor
        for i, item in enumerate(batch):
            y_, x_, spk_ = item['y'], item['x'], item['spk']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            spk.append(spk_)
            if "pit" in item.keys():
                pit, eng, dur = item["pit"], item["eng"], item["dur"]
                ## Sequential length for psd
                psd_lengths = []
                psd_lengths.append(pit.shape[-1])
                pits[i, :pit.shape[-1]] = pit[0]
                engs[i, :eng.shape[-1]] = eng[0]
                durs[i, :dur.shape[-1]] = dur[0]
            if "melstyle" in item.keys():
                melstyle = item["melstyle"]
                melstyle_lengths = []
                melstyle_lengths.append(melstyle.shape[-1])
                melstyles[i, :, :melstyle.shape[2]] = melstyle[0]
                melstyle_lengths = torch.LongTensor(melstyle_lengths)
            if "emo_label" in item.keys():
                emo_ = item["emo_label"]
                if len(emo.shape) == 2:
                    emo[i, :] = emo_  # for gradTTS
                else:
                    emo[i] = emo_
            if "emo" in item.keys():
                emo_emb_ = item["emo"]
                emo_emb[i, :] = emo_emb_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk = torch.cat(spk, dim=0)
        # emo = torch.FloatTensor(emo)

        if "melstyle" in batch[0].keys():
            return {'x': x,
                    'x_lengths': x_lengths,
                    'y': y,
                    'y_lengths': y_lengths,
                    'spk': spk,
                    "emo_label": emo,
                    "melstyle": melstyles,
                    "melstyle_lengths": melstyle_lengths
                    }
        elif "pit" in batch[0].keys():
            return {'x': x,
                    'x_lengths': x_lengths,
                    'y': y,
                    'y_lengths': y_lengths,
                    'spk': spk,
                    "emo_label": emo,
                    "pit": pits,
                    "eng": engs,
                    "dur": durs,
                    "psd_lengths": psd_lengths
                    }
        else:
            return {'x': x,
                    'x_lengths': x_lengths,
                    'y': y,
                    'y_lengths': y_lengths,
                    'spk': spk,
                    "emo": emo_emb,
                    }


def get_mel(filepath,
            spk=None,
            preprocessed_path=None,
            ):
    mel_path = os.path.join(
        preprocessed_path,
        "mel",
        "{}-mel-{}.npy".format(spk, filepath),
    )
    mel = torch.from_numpy(np.load(mel_path))
    mel = torch.transpose(mel, 0, 1)
    return mel