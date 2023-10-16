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

#HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
#HIFIGAN_CHECKPT = './checkpts/hifigan.pt'
HIFIGAN_CONFIG = './checkpts/config.json'
HIFIGAN_CHECKPT = './checkpts/generator_v3'

def inference(params, chk_pt, syn_txt, time_steps, spk):
    print('Initializing Grad-TTS...')
    generator = GradTTS(len(symbols) + 1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
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
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=time_steps, temperature=1.5,
                                                   stoc=False, spk=spk, length_scale=0.91)
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')

            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

            write(f'./out/sample_{i}.wav', 22050, audio)

    print('Done. Check out `out` folder for samples.')


def inference_emo_interpolation(params, chk_pt, syn_txt, time_steps, spk, emo1, emo2, out_dir):
    generator = CondGradTTS(len(symbols) + 1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(chk_pt, map_location=lambda loc, storage: loc))
    _ = generator.cuda().eval()
    #print(f'Number of parameters: {generator.nparams}')

    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    c = torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)
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
            # Before Interpolation
            y_enc1, y_dec1, attn1 = generator.forward(x, x_lengths, n_timesteps=time_steps, temperature=1.5,
                                                   stoc=False, spk=spk, length_scale=0.91, emo=emo1)
            y_enc2, y_dec2, attn2 = generator.forward(x, x_lengths, n_timesteps=time_steps, temperature=1.5,
                                                   stoc=False, spk=spk, length_scale=0.91, emo=emo2)
            audio1 = (vocoder.forward(y_dec1).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            audio2 = (vocoder.forward(y_dec2).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

            # After Interpolation
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=time_steps, temperature=1.5,
                                                   stoc=False, spk=spk, length_scale=0.91, emo=emo1, emo2=emo2)
            audio_interpolate = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

            #t = (dt.datetime.now() - t).total_seconds()
            #print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')

            if not os.path.isdir(out_dir):
                Path(out_dir).mkdir(exist_ok=True)

            write(f'./{out_dir}/sample_{i}_emo1.wav', 22050, audio1)
            write(f'./{out_dir}/sample_{i}_emo2.wav', 22050, audio2)
            write(f'./{out_dir}/sample_{i}_emo12.wav', 22050, audio_interpolate)

    print('Done. Check out `out` folder for samples.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=False,
                        default="resources/filelists/synthesis.txt",
                        help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, required=False,
                        default="/home/rosen/Project/Speech-Backbones/Grad-TTS/logs/ESD_gradtts_local/grad_11.pt",
                        #default="/home/rosen/Project/Speech-Backbones/Grad-TTS/logs/ESD_gradtts/grad_27.pt",
                        help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=50, help='number of timesteps of reverse diffusion')
    parser.add_argument('-s', '--speaker_id', type=int, required=False, default=19, help='speaker id for multispeaker model')
    args = parser.parse_args()
    
    if not isinstance(args.speaker_id, type(None)):
        assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
        spk = torch.LongTensor([args.speaker_id]).cuda()
    else:
        spk = None


    # Inference speech from text and spkid, given checkpoint
    ## Input
    chk_pt = args.checkpoint

    syn_txt = args.file
    time_steps = args.timesteps

    if False:
        ## Output -> out
        inference(params, chk_pt, syn_txt, time_steps, spk)
    if True:
        # Inference speech with interpolation

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
        inference_emo_interpolation(params, chk_pt, syn_txt, time_steps, spk, emo1, emo2, out_dir)

        # 2. Interpolate z


