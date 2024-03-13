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
#HIFIGAN_CONFIG = './checkpts/config.json'
#HIFIGAN_CHECKPT = './checkpts/generator_v3'

HIFIGAN_CONFIG = './checkpts/hifigan-config.json' # ./checkpts/config.json
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'



def get_model(configs, model="gradtts_lm"):
    """
    Get Model

    """
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
    sample_channel_n = int(model_config["decoder"]["sample_channel_n"])

    ### unet
    unet_type = model_config["unet"]["unet_type"]
    att_type = model_config["unet"]["att_type"]
    att_dim = model_config["unet"]["att_dim"]
    heads = model_config["unet"]["heads"]
    p_uncond = model_config["unet"]["p_uncond"]

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
    elif model == "gradtts_cross":
        return CondGradTTS(nsymbols,
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
                            sample_channel_n,
                            beta_min,
                            beta_max,
                            pe_scale,
                            unet_type,
                            att_type,
                            att_dim,
                            heads,
                            p_uncond)


def inference(configs,
              model_name,
              chk_pt,
              syn_txt=None,
              time_steps=50,
              spk=None,
              emo_label=None,
              melstyle=None,
              out_dir=None,
              guidence_strength=3.0
              ):

    # Get model
    print('Initializing Grad-TTS...')
    generator = get_model(configs, model_name)
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
            if model_name == "gradtts_lm":
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
            elif model_name == "gradtts_cross":
                y_enc, y_dec, attn = generator.forward(x,
                                                       x_lengths,
                                                       n_timesteps=time_steps,
                                                       temperature=1.5,
                                                       stoc=True,
                                                       length_scale=0.91,
                                                       spk=spk,
                                                       emo_label=emo_label,
                                                       melstyle=melstyle,
                                                       guidence_strength=guidence_strength
                                                       )
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')
            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            write(f'{out_dir}/sample_{i}.wav', 22050, audio)
    print('Done. Check out `out` folder for samples.')


def inference_interp(
        configs,
        model_name,
        chk_pt,
        syn_txt=None,
        time_steps=50,
        spk=None,
        emo_label1=None,
        melstyle1=None,
        pitch1=None,
        emo_label2=None,
        melstyle2=None,
        pitch2=None,
        out_dir=None,
        interp_type="temp",
        mask_time_step=None,
        mask_all_layer=True,
        temp_mask_value=0,
        guidence_strength=3.0
):

    # Get model
    print('Initializing Grad-TTS...')
    generator = get_model(configs, model_name)
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
            if model_name == "gradtts_lm":
                y_enc, y_dec, attn = generator.forward(x,
                                                       x_lengths,
                                                       n_timesteps=time_steps,
                                                       temperature=1.5,
                                                       stoc=True,
                                                       spk=spk,
                                                       length_scale=0.91,
                                                       emo_label=emo_label1,
                                                       #ref_speech=None,
                                                       melstyle=melstyle1,
                                                       )
                y_dec = y_dec.transpose(1, 2)
            elif model_name == "gradtts_cross":
                y_enc, y_dec, attn = generator.reverse_diffusion_interp(
                    x,
                    x_lengths,
                    n_timesteps=time_steps,
                    temperature=1.5,
                    stoc=True,
                    length_scale=0.91,
                    spk=spk,
                    emo_label1=emo_label1,
                    melstyle1=melstyle1,
                    pitch1=pitch1,
                    emo_label2=emo_label2,
                    melstyle2=melstyle2,
                    pitch2=pitch2,
                    interp_type=interp_type,
                    mask_time_step=mask_time_step,
                    mask_all_layer=mask_all_layer,
                    temp_mask_value=temp_mask_value,
                    guidence_strength=guidence_strength
                )
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
    # constant parameter
    emo_num_dict = {
        "Angry": 0,
        "Surprise": 1,
        "Sad": 2,
        "Neutral": 3,
        "Happy": 4
    }
    emo_melstyle_dict = {
        "Angry": "0015_000415.npy",  # Tom could hardly speak for laughing
        "Surprise": "0015_001465.npy",
        "Sad": "0015_001115.npy",   # Tom could hardly speak for laughing
        "Neutral": "0015_000065.npy",
        "Happy": "0015_000765.npy"
    }
    # used for extract pitch (Phoneme average)
    psd_dict = {
        "Angry": "0015-pitch-0015_000415.npy",  # Tom could hardly speak for laughing
        "Surprise": "0015-pitch-0015_001465.npy",
        "Sad": "0015-pitch-0015_001115.npy",  # Tom could hardly speak for laughing
        "Neutral": "0015-pitch-0015_000065.npy",
        "Happy": "0015-pitch-0015_000765.npy"
    }
    # used for extract pitch (No phoneme average)
    wav_dict = {
        "Angry": "0015_000415.wav",  # Tom could hardly speak for laughing
        "Surprise": "0015_001465.wav",
        "Sad": "0015_001115.wav",  # Tom could hardly speak for laughing
        "Neutral": "0015_000065.npy",
        "Happy": "0015_000765.npy"
    }

    logs_dir_par = "/home/rosen/Project/Speech-Backbones/GradTTS/logs/"
    #logs_dir = logs_dir_par + "gradtts_crossSelf_v2/"
    logs_dir = logs_dir_par + "gradtts_crossSelf_puncond_n1_neworder_fixmask/"

    config_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/config/ESD"
    melstyle_dir = "/home/rosen/Project/FastSpeech2/preprocessed_data/ESD/emo_reps"
    psd_dir = "/home/rosen/Project/FastSpeech2/preprocessed_data/ESD/pitch"
    wav_dir = "/home/rosen/Project/FastSpeech2/ESD/16k_wav"


    ## INPUT
    ### General
    checkpoint = "grad_54.pt"
    chk_pt = logs_dir + checkpoint
    syn_txt = "resources/filelists/synthesis1.txt"
    time_steps = 100
    model_name = "gradtts_cross"

    speaker_id = 15
    spk_tensor = torch.LongTensor([speaker_id]).cuda() if not isinstance(speaker_id, type(None)) else None
    ### Style
    emo_label = "Angry"
    emo_label_tensor = torch.LongTensor(get_emo_label(emo_label)).cuda()
    melstyle_tensor = torch.from_numpy(np.load(
        melstyle_dir + "/" + emo_melstyle_dict[emo_label])).cuda()
    melstyle_tensor = melstyle_tensor.unsqueeze(0).transpose(1, 2)
    guidence_strength = 3.0

    preprocess_config = yaml.load(
        open(config_dir + "/preprocess_gradTTS.yaml", "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(
        config_dir + "/model_gradTTS.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        config_dir + "/train_gradTTS.yaml", "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    chpt_n = checkpoint.split(".")[0].split("_")[1]

    # Interpolation input
    sad1 = "0019_001115.npy"
    ang1 = "0019_000508.npy"
    melstyle_sad = torch.from_numpy(np.load(melstyle_dir + "/" + sad1)).unsqueeze(0).transpose(1, 2).cuda()
    melstyle_ang = torch.from_numpy(np.load(melstyle_dir + "/" + ang1)).unsqueeze(0).transpose(1, 2).cuda()

    emo_label1 = "Sad"
    emo_label2 = "Angry"
    emo_label_sad = torch.LongTensor(get_emo_label(emo_label1)).cuda()
    emo_label_ang = torch.LongTensor(get_emo_label(emo_label2)).cuda()

    # psd input (N,)
    """
    wav_sad = torch.from_numpy(np.load(
        psd_dir + "/" + psd_dict["Sad"])).cuda()
    wav_ang = torch.from_numpy(np.load(
        psd_dir + "/" + psd_dict["Angry"])).cuda()
    """
    wav_sad = wav_dir + "/" + wav_dict["Sad"]
    wav_ang = wav_dir + "/" + wav_dict["Angry"]

    NO_INTERPOLATION = False
    INTERPOLATE_INFERENCE_SIMP = False
    # Temp
    INTERPOLATE_INFERENCE_TEMP = False  # synthesize
    INTERPOLATE_INFERENCE_TEMP_EMO_DETECT = False # evaluation emoTempInterp

    # Freq
    INTERPOLATE_INFERENCE_FREQ = True   # synthesize
    INTERPOLATE_INFERENCE_FREQ_PSD_MATCH = False  # evaluation emoFreqInterp

    # inference without interpolation
    if NO_INTERPOLATION:
        # INPUT
        emo_labels = ["Angry", "Sad", "Happy"]
        guidence_strengths = [0.0, 0.5, 1.0, 3.0, 5.0]  # 0: only cond_diff

        for emo_label in emo_labels:
            for guidence_strength in guidence_strengths:
                emo_label_tensor = torch.LongTensor(get_emo_label(emo_label)).cuda()
                melstyle_tensor = torch.from_numpy(np.load(
                    melstyle_dir + "/" + emo_melstyle_dict[emo_label])).cuda()
                melstyle_tensor = melstyle_tensor.unsqueeze(0).transpose(1, 2)

                # OUTPUT
                out_dir = f"{logs_dir}chpt{chpt_n}_time{time_steps}_spk{speaker_id}_emo{emo_label}_w{guidence_strength}"
                if not os.path.isdir(out_dir):
                    Path(out_dir).mkdir(exist_ok=True)

                inference(configs,
                          model_name,
                          chk_pt,
                          syn_txt=syn_txt,
                          time_steps=time_steps,
                          spk=spk_tensor,
                          emo_label=emo_label_tensor,
                          melstyle=melstyle_tensor,
                          out_dir=out_dir,
                          guidence_strength=guidence_strength
                          )

    if INTERPOLATE_INFERENCE_SIMP:
        # input
        interp_type = "simp"
        # output
        out_dir = f"{logs_dir}chpt{chpt_n}_time{time_steps}_interp{interp_type}_spk{speaker_id}_{emo_label1}_{emo_label2}"
        if not os.path.isdir(out_dir):
            Path(out_dir).mkdir(exist_ok=True)

        inference_interp(
            configs,
            model_name,
            chk_pt,
            syn_txt=syn_txt,
            time_steps=time_steps,
            spk=spk_tensor,
            emo_label1=emo_label_sad,
            melstyle1=melstyle_sad,
            emo_label2=emo_label_ang,
            melstyle2=melstyle_ang,
            out_dir=out_dir,
            interp_type=interp_type,
            guidence_strength=guidence_strength

        )

        # 1. interpolate score
        # inference_emo_interpolation(params, chk_pt, syn_txt, time_steps, spk, emo1, emo2, out_dir)
        """
        # between inner-emotion
        # between inner-emotion and different intensity
        emo1 = torch.from_numpy(np.load(emo_emb_dir + sad1).squeeze(0)).cuda()
        emo2 = torch.from_numpy(np.load(emo_emb_dir + ang1).squeeze(0)).cuda()

        # 1. interpolate score
        #inference_emo_interpolation(params, chk_pt, syn_txt, time_steps, spk, emo1, emo2, out_dir)

        # 2. Interpolate z
        """

    if INTERPOLATE_INFERENCE_TEMP:
        # input
        interp_type = "temp"
        mask_time_step = int(time_steps / 1.0)
        mask_all_layer = True
        temp_mask_value = 0
        guidence_strength = 3.0

        # output
        if mask_all_layer:
            mask_range_tag = "maskAllLayers"
        else:
            mask_range_tag = "maskFirstLayers"
        out_dir = (f"{logs_dir}chpt{chpt_n}_time{time_steps}_interp{interp_type}_spk{speaker_id}_{emo_label1}_"
                   f"{emo_label2}_{mask_range_tag}_timeSplit_{guidence_strength}_test")
        if not os.path.isdir(out_dir):
            Path(out_dir).mkdir(exist_ok=True)

        inference_interp(
            configs,
            model_name,
            chk_pt,
            syn_txt=syn_txt,
            time_steps=time_steps,
            spk=spk_tensor,
            emo_label1=emo_label_sad,
            melstyle1=melstyle_sad,
            emo_label2=emo_label_ang,
            melstyle2=melstyle_ang,
            out_dir=out_dir,
            interp_type=interp_type,
            mask_time_step=mask_time_step,
            mask_all_layer=mask_all_layer,
            temp_mask_value=temp_mask_value,
            guidence_strength=guidence_strength
        )

    if INTERPOLATE_INFERENCE_FREQ:
        # input
        interp_type = "freq"
        mask_time_step = int(time_steps / 1.0)
        mask_all_layer = True
        temp_mask_value = 0
        guidence_strength = 3.0

        # output
        out_dir = f"{logs_dir}chpt{chpt_n}_time{time_steps}_interp{interp_type}_spk{speaker_id}_{emo_label1}_{emo_label2}_v3_maskFirstLayers_timeSplit"
        if not os.path.isdir(out_dir):
            Path(out_dir).mkdir(exist_ok=True)

        inference_interp(
            configs,
            model_name,
            chk_pt,
            syn_txt=syn_txt,
            time_steps=time_steps,
            spk=spk_tensor,
            emo_label1=emo_label_sad,
            melstyle1=melstyle_sad,
            pitch1=wav_sad,
            emo_label2=emo_label_ang,
            melstyle2=melstyle_ang,
            pitch2=wav_ang,
            out_dir=out_dir,
            interp_type=interp_type,
            mask_time_step=mask_time_step,
            mask_all_layer=mask_all_layer,
            temp_mask_value=temp_mask_value,
            guidence_strength=guidence_strength
        )

