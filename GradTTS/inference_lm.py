# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import datetime as dt
import os

import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS, CondGradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
from pathlib import Path
from model import GradTTS, CondGradTTS, CondGradTTSLDM
import yaml

#HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
#HIFIGAN_CHECKPT = './checkpts/hifigan.pt'
HIFIGAN_CONFIG = './checkpts/config.json'
HIFIGAN_CHECKPT = './checkpts/generator_v3'


def get_model(configs, model="gradtts_lm"):
    preprocess_config, model_config, train_config = configs

    # parameter
    add_blank = preprocess_config["feature"]["add_blank"]
    nsymbols = len(symbols) + 1 if add_blank else len(symbols)
    n_spks = int(preprocess_config["feature"]["n_spks"])
    n_feats = int(preprocess_config["feature"]["n_feats"])

    # model
    ## Encoder
    spk_emb_dim = int(model_config["spk_emb_dim"])
    emo_emb_dim = int(model_config["emo_emb_dim"])
    n_enc_channels = int(model_config["encoder"]["n_enc_channels"])
    filter_channels = int(model_config["encoder"]["filter_channels"])
    filter_channels_dp = int(model_config["encoder"]["filter_channels_dp"])
    n_heads = int(model_config["encoder"]["n_heads"])
    n_enc_layers = int(model_config["encoder"]["n_enc_layers"])
    enc_kernel = int(model_config["encoder"]["enc_kernel"])
    enc_dropout = float(model_config["encoder"]["enc_dropout"])
    window_size = int(model_config["encoder"]["window_size"])
    length_scale = float(model_config["encoder"]["length_scale"])

    ## Decoder
    dec_dim = int(model_config["decoder"]["dec_dim"])
    beta_min = float(model_config["decoder"]["beta_min"])
    beta_max = float(model_config["decoder"]["beta_max"])
    pe_scale = int(model_config["decoder"]["pe_scale"])
    stoc = model_config["decoder"]["stoc"]
    temperature = float(model_config["decoder"]["temperature"])
    n_timesteps = int(model_config["decoder"]["n_timesteps"])

    ### unet
    unet_type = model_config["unet"]["unet_type"]
    att_type = model_config["unet"]["att_type"]

    if model == "gradtts_lm":
        return CondGradTTSLDM(
            nsymbols,
            n_spks,
            spk_emb_dim,
            emo_emb_dim,
            n_enc_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_enc_layers,
            enc_kernel,
            enc_dropout,
            window_size,
            n_feats,
            dec_dim,
            beta_min,
            beta_max,
            pe_scale,
            att_type)
    else:
        return 0
def inference(configs,
              chk_pt,
              syn_txt=None,
              time_steps=50,
              spk=None,
              emo_label=None,
              melstyle=None,
              out_dir=None):

    # get model

    print('Initializing Grad-TTS...')
    generator = get_model(configs, "gradtts_lm")
    generator.load_state_dict(torch.load(chk_pt, map_location=lambda loc, storage: loc))
    _ = generator.cuda().eval()

    print(f'Number of parameters: {generator.nparams}')
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    with open(syn_txt, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')

    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'Synthesizing {i} text...', end=' ')
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()

            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x,
                                                   x_lengths,
                                                   n_timesteps=time_steps,
                                                   temperature=1.5,
                                                   stoc=True,
                                                   spk=spk,
                                                   length_scale=0.91,
                                                   emo_label=emo_label,
                                                   #ref_speech=None,
                                                   melstyle=melstyle,
                                                   )
            y_dec = y_dec.transpose(1, 2)
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')
            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            write(f'{out_dir}/sample_{i}.wav', 22050, audio)
    print('Done. Check out `out` folder for samples.')


def get_emo_label(emo: str, emo_value=1):
    emo_emb_dim = len(emo_num_dict.keys())
    emo_label = [[0] * emo_emb_dim]
    emo_label[0][emo_num_dict[emo]] = emo_value
    return emo_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=False,
                        default="resources/filelists/synthesis1.txt",
                        help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint',
                        type=str,
                        required=False,
                        default="grad_57.pt",
                        help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False,
                        default=100, help='number of timesteps of reverse diffusion')
    parser.add_argument('-s', '--speaker_id', type=int, required=False,
                        default=15, help='speaker id for multispeaker model')
    parser.add_argument('-e', '--emo_label', type=str, required=False,
                        default="Sad")

    args = parser.parse_args()

    # constant parameter
    emo_num_dict = {
        "Angry": 0,
        "Surprise": 1,
        "Sad": 2,
        "Neutral": 3,
        "Happy": 4
    }
    emo_melstyle_dict = {
        "Angry": "0019_000508.npy",
        "Surprise": "",
        "Sad": "0019_001115.npy",
        "Neutral": "",
        "Happy": ""
    }
    logs_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/logs/crossatt_diffuser_nopriorloss/"
    config_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/config/ESD"
    melstyle_dir = "/home/rosen/Project/FastSpeech2/preprocessed_data/ESD/emo_reps"

    # Input value
    spk_tensor = torch.LongTensor([args.speaker_id]).cuda() if not isinstance(args.speaker_id, type(None)) else None
    ## style
    emo_label = args.emo_label
    emo_label_tensor = torch.LongTensor(get_emo_label(emo_label)).cuda()
    melstyle_tensor = torch.from_numpy(np.load(
        melstyle_dir + "/" + emo_melstyle_dict[emo_label])).cuda()
    melstyle_tensor = melstyle_tensor.unsqueeze(0).transpose(1, 2)

    chk_pt = logs_dir + args.checkpoint
    ## content
    syn_txt = args.file
    time_steps = args.timesteps

    preprocess_config = yaml.load(
        open(config_dir + "/preprocess_gradTTS.yaml", "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(
        config_dir + "/model_gradTTS.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        config_dir + "/train_gradTTS.yaml", "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # output
    chpt_n = args.checkpoint.split(".")[0].split("_")[1]
    out_dir = f"{logs_dir}chpt{chpt_n}_time{time_steps}_spk{args.speaker_id}_emo{emo_label}"
    if not os.path.isdir(out_dir):
        Path(out_dir).mkdir(exist_ok=True)

    INTERPOLATE_INFERENCE = False
    # inference without interpolation
    if not INTERPOLATE_INFERENCE:
        inference(configs,
                  chk_pt,
                  syn_txt=syn_txt,
                  time_steps=time_steps,
                  spk=spk_tensor,
                  emo_label=emo_label_tensor,
                  melstyle=melstyle_tensor,
                  out_dir=out_dir
                  )

    if INTERPOLATE_INFERENCE:
        # Inference speech with interpolationks
        emo_emb_dir = "/home/rosen/Project/FastSpeech2/preprocessed_data/ESD/iiv_reps/"
        time_steps = 100
        # between inter-emotion
        sad1 = "0019_001115.npy"
        ang1 = "0019_000508.npy"
        chk_pt = "/home/rosen/Project/Speech-Backbones/Grad-TTS/logs/ESD_gradtts/grad_22.pt"

        out_dir = f"out/chpt{time_steps}"
        # between inner-emotion
        # between inner-emotion and different intensity
        emo1 = torch.from_numpy(np.load(emo_emb_dir + sad1).squeeze(0)).cuda()
        emo2 = torch.from_numpy(np.load(emo_emb_dir + ang1).squeeze(0)).cuda()

        # 1. interpolate score
        #inference_emo_interpolation(params, chk_pt, syn_txt, time_steps, spk, emo1, emo2, out_dir)

        # 2. Interpolate z


