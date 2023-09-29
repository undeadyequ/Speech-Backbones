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
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS, CondDiffusion
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN


HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'

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


def inference_emo_interpolation(params, chk_pt, syn_txt, time_steps, spk, emo1, emo2):
    generator = CondDiffusion(len(symbols) + 1, params.n_spks, params.spk_emb_dim,
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
                                                   stoc=False, spk=spk, length_scale=0.91, emo1=emo1)
            y_enc2, y_dec2, attn2 = generator.forward(x, x_lengths, n_timesteps=time_steps, temperature=1.5,
                                                   stoc=False, spk=spk, length_scale=0.91, emo2=emo2)
            audio1 = (vocoder.forward(y_dec1).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            audio2 = (vocoder.forward(y_dec2).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

            # After Interpolation
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=time_steps, temperature=1.5,
                                                   stoc=False, spk=spk, length_scale=0.91, emo1=emo1, emo2=emo2)
            audio_interpolate = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

            #t = (dt.datetime.now() - t).total_seconds()
            #print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')
            write(f'./before_interploate/emo1_sample_{i}.wav', 22050, audio1)
            write(f'./before_interploate/emo2_sample_{i}_emo2.wav', 22050, audio2)
            write(f'./after_interploate/interpolate_sample_{i}.wav', 22050, audio_interpolate)

    print('Done. Check out `out` folder for samples.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')
    parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None, help='speaker id for multispeaker model')
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
    ## Output -> out
    inference(params, chk_pt, syn_txt, time_steps, spk)

    # Inference speech with interpolation
    emo_emb_dir = ""
    emo1_id = "?"
    emo2_id = "?"
    emo1 = torch.LongTensor(emo_emb_dir + emo1_id)
    emo2 = torch.LongTensor(emo_emb_dir + emo2_id)
    inference_emo_interpolation(params, chk_pt, syn_txt, time_steps, spk, emo1, emo2)




