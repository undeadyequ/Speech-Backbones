from distutils.command.config import config

import torch
from GradTTS.read_model import get_model, get_vocoder


import json
import datetime as dt
import os
import sys

import numpy as np
from scipy.io.wavfile import write
import torch
from typing import Union

from text import text_to_sequence, cmudict, text_to_arpabet
from text.symbols import symbols
from utils import intersperse

sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
from pathlib import Path
from model import GradTTS, CondGradTTS, CondGradTTSLDM
import yaml
from utils import get_emo_label
from model.maskCreation import create_p2p_mask, DiffAttnMask
from GradTTS.exp.utils import get_ref_pRange, get_tgt_pRange
from GradTTS.data import get_mel
from GradTTS.utils import parse_filelist
from GradTTS.util.util_config import convert_to_generator_input

def evaluate_prosody_similarity(
        style_mult,
        synText_mult,
        model_name1,
        model_name2,
        configs1,
        configs2,
        vocoder_name,
        vocoder_config,
        out_dir,
        vis_only,
):
    """
    1. Get input
    - Style: emotion lable, Reference speech, speaker
        - one style  : tuple(emo, ref, spk)
        - multi sytle:   {emo: [(ref, spk))]}
    - SynText
        - one text: "t1"
        - multi text: ["t1", "t2"]
    2. Set model
    - Generator : name, configs (pre/model/train)
    - includes gradTTS w/o attnMask, w/o noise manipulation
    - Vocoder: name, vocder_config
    3. Synthesize speech with given input/model and save them in folder (inference.py)
    4. Compute evaluate matrix and save it in json file
    5. Visualize from json file

    Returns:
    """

    # 1. Get input
    syntext_list = synText_mult

    # 2. Set model
    vocoder = get_vocoder()

    # 3. Synthesize speech with given input/model and save them in folder (inference.py)

    with torch.no_grad():
        for model_n, configs in ([model_name1, configs1], [model_name2, configs2]):
            generator = get_model(configs, model_n)

            for i, text in enumerate(syntext_list):
                x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmudict), len(symbols))).cuda()[None]
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                for style in style_mult:
                    generator_input = convert_to_generator_input(model_n, text, style)
                    y_enc, y_dec, attn = generator.forward(generator_input)
                    y_dec = y_dec.transpose(1, 2)
                    # vocoding
                    t = (dt.datetime.now() - t).total_seconds()
                    print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')
                    audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                    audio_id = "_".join(style) + f"_{i}"

                    out_sub_dir = os.path.join(out_dir, model_n)
                    write(f'{out_dir}/{audio_id}.wav', 22050, audio)


    # 4. Compute evaluate matrix and save it in json file
    for speech in os.listdir(out_dir):
        speech_f = os.path.join(out_dir, speech)



    # 5. Visualize from json file


