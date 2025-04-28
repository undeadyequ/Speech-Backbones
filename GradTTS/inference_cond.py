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
from model.util_maskCreation import create_p2p_mask, DiffAttnMask
from GradTTS.exp.utils import get_ref_pRange, get_tgt_pRange
from GradTTS.data import get_mel
from GradTTS.utils import parse_filelist

#HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
#HIFIGAN_CHECKPT = './checkpts/hifigan.pt'
#HIFIGAN_CONFIG = './checkpts/config.json'
#HIFIGAN_CHECKPT = './checkpts/generator_v3'

HIFIGAN_CONFIG = '/home/rosen/Project/Speech-Backbones/GradTTS/hifi-gan/checkpts/hifigan-config.json' # ./checkpts/config.json
HIFIGAN_CHECKPT = '/home/rosen/Project/Speech-Backbones/GradTTS/checkpts/hifigan.pt'

cmu = cmudict.CMUDict('/home/rosen/Project/Speech-Backbones/GradTTS/resources/cmu_dictionary')


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

    """
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
    """
    if model == "gradtts_cross":
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

def load_model_state(checkpoint: Union[str, Path], model: torch.nn.Module, ngpu=1):
    ckpt_states = torch.load(
        checkpoint,
        map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu",
    )

    if "model_state_dict" in ckpt_states:
        model.load_state_dict(
            ckpt_states["model_state_dict"])
    else:  # temp use
        states = torch.load(
            checkpoint,
            map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu",
        )
        model.load_state_dict(states)


def inference(configs,
              model_name,
              chk_pt,
              syn_txt=None,
              time_steps=50,
              spk=None,
              emo_label=None,
              melstyle=None,
              out_dir=None,
              guidence_strength=3.0,
              stoc=False,
              output_pre="sample_"
              ):
    """
    Basic inference given txt, cond(spk, emo_label, melstyle)
    """

    # Get model
    print('Initializing Grad-TTS...')
    generator = get_model(configs, model_name)

    load_model_state(chk_pt, generator)
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
                                                       stoc=stoc,
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
                                                       temperature=1.5,    # 1.5
                                                       stoc=stoc,
                                                       length_scale=0.91,  # 0.91
                                                       spk=spk,
                                                       emo_label=emo_label,
                                                       melstyle=melstyle,
                                                       guidence_strength=guidence_strength
                                                       )
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')
            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            write(f'{out_dir}/{output_pre}{i}.wav', 22050, audio)
    print('Done. Check out `out` folder for samples.')


def inference_p2p_transfer_by_attnMask(
        configs,
        model_name,
        chk_pt,
        syn_txt=None,
        time_steps=50,

        spk1=None,
        emo_label1=None,
        melstyle1=None,
        attn_hardMask1=None,

        spk2=None,
        emo_label2=None,
        melstyle2=None,
        attn_hardMask2=None,

        out_dir=None,
        guidence_strength=3.0,
        stoc=False,
        output_pre="sample_",

        ref_p_start=5,
        ref_p_end=8,
        tgt_p_start=8,
        tgt_p_end=9,
        p2p_mode="noise"
):
    """
    P2P transfer by adding cross attention mask where p2p area setting to 1.0

    1. start and end phoneme of target speech
    tgt_p_start = 8  # still 7 - 10 (8 9: till)  forest  18 - 26 (rest: 21 25)
    tgt_p_end = 9  # ?? get it from x with 148? ??

    2. start and end phoneme of reference speech
    ref_p_start=5,
    ref_p_end=8,
    """

    # Create p2p mask
    # Get model
    print('Initializing Grad-TTS...')
    generator = get_model(configs, model_name)

    load_model_state(chk_pt, generator)
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

    # Get frame-level transfering phoneme start/end on reference speech by textGrid
    ref_pFrame_range = get_ref_pRange(ref_p_start, ref_p_end, ref2_id)

    # Get phoneme-level taget start/edn on target speech  <- big problem ?
    tgt_p_start = tgt_p_start * 2
    tgt_p_end = tgt_p_end * 2
    tgt_pRange = (tgt_p_start, tgt_p_end)  # because adding 148 interpolated with x exp. 40 - 60  # should

    if p2p_mode == "noise":
        p2p_mask = None
        ref2_y = get_mel(ref2_id, ref2_id.split("_")[0], mel_dir).cuda()
        ref2_start_p = ref_pFrame_range[0]
        ref2_end_p = ref_pFrame_range[1]
    elif p2p_mode == "crossAttn":
        ref2_y = None
        ref2_start_p = None
        ref2_end_p = None
    else:
        raise IOError("p2p model should be noise or crossAttn")

    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'Synthesizing {i} text...', end=' ')
            a = intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
            t = dt.datetime.now()

            # create p2p mask
            p2p_mask1 = create_p2p_mask(
                tgt_size=x.shape[1],
                ref_size=melstyle1.shape[2],
                dims=4,
                tgt_range=tgt_pRange,
                ref_range=ref_pFrame_range
            )
            p2p_mask1 = p2p_mask1.cuda()
            p2p_mask2 = 1 - p2p_mask1
            ############# CHECK2.1 start/end of ref and tgt (grid=0 circled by extra 2dims with 1) ###################
            print("p2p grid mask with 2 extra value on 1/2 dims \n")
            print(p2p_mask1[0, 0, tgt_pRange[0] - 2: tgt_pRange[1] + 2,
                  ref_pFrame_range[0] - 2:ref_pFrame_range[1] + 2])  # masked=0, unmasked=1

            y_enc, y_dec, attn, attn_hardMask = generator.reverse_diffusion_mix(
                x,
                x_lengths,
                n_timesteps=time_steps,
                temperature=1.5,    # 1.5
                stoc=stoc,
                length_scale=0.91,   # 0.91
                mix_mode=p2p_mode,

                spk1=spk1,
                emo_label1=emo_label1,
                melstyle1=melstyle1,
                attn_hardMask1=p2p_mask1,

                spk2=spk2,
                emo_label2=emo_label2,
                melstyle2=melstyle2,
                attn_hardMask2=p2p_mask2,

                guidence_strength=guidence_strength,
                #ref2_y=ref2_y,
                ref2_start_p=ref2_start_p,
                ref2_end_p=ref2_end_p,
                tgt_start_p=tgt_p_start,
                tgt_end_p=tgt_p_end
            )
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')
            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            write(f'{out_dir}/{output_pre}{i}.wav', 22050, audio)
    print('Done. Check out `out` folder for samples.')
    return y_dec, attn_hardMask


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
        guidence_strength=3.0,
        stoc=False
):

    # Get model
    print('Initializing Grad-TTS...')
    generator = get_model(configs, model_name)
    load_model_state(chk_pt, generator)

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
    cmu = cmudict.CMUDict('/home/rosen/Project/Speech-Backbones/GradTTS/resources/cmu_dictionary')

    #with torch.no_grad():
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
                                                   stoc=stoc,
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
                stoc=stoc,
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
        #y_dec = y_dec.numpy()
        #audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
        audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).detach().numpy() * 32768).astype(np.int16)
        write(f'{out_dir}/sample_{i}.wav', 22050, audio)
    print('Done. Check out `out` folder for samples.')


if __name__ == '__main__':
    # constant parameter
    from const_param import emo_num_dict, emo_melstyle_dict, psd_dict, wav_dict, emo_melstyle_dict2
    from const_param import logs_dir_par, logs_dir, config_dir, melstyle_dir, psd_dir, wav_dir

    ## INPUT
    ### General
    checkpoint = "grad_400.pt"  # grad_65
    chk_pt = logs_dir + "models/" + checkpoint
    syn_txt = "resources/filelists/synthesis1_closetext_onlyOne.txt"
    time_steps = 50
    model_name = "gradtts_cross"
    speaker_id = 19 # 15
    spk_tensor = torch.LongTensor([speaker_id]).cuda() if not isinstance(speaker_id, type(None)) else None

    ### Style
    emo_label = "Angry"
    emo_label_tensor = torch.LongTensor(get_emo_label(emo_label, emo_num_dict)).cuda()
    melstyle_tensor = torch.from_numpy(np.load(
        melstyle_dir + "/" + emo_melstyle_dict[emo_label])).cuda()
    melstyle_tensor = melstyle_tensor.unsqueeze(0).transpose(1, 2)
    guidence_strength = 3.0
    stoc = False

    preprocess_config = yaml.load(
        open(config_dir + "/preprocess_gradTTS.yaml", "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(
        config_dir + "/model_gradTTS_v3.yaml", "r"), Loader=yaml.FullLoader)
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
    emo_label_sad = torch.LongTensor(get_emo_label(emo_label1, emo_num_dict)).cuda()
    emo_label_ang = torch.LongTensor(get_emo_label(emo_label2, emo_num_dict)).cuda()

    # psd input (N,)
    """
    wav_sad = torch.from_numpy(np.load(
        psd_dir + "/" + psd_dict["Sad"])).cuda()
    wav_ang = torch.from_numpy(np.load(
        psd_dir + "/" + psd_dict["Angry"])).cuda()
    """
    wav_sad = wav_dir + "/" + wav_dict["Sad"]
    wav_ang = wav_dir + "/" + wav_dict["Angry"]

    # inference
    mel_dir = "/home/rosen/Project/FastSpeech2/preprocessed_data/ESD"
    ref_indx_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/resources/filelists"

    NO_INTERPOLATION = False
    INTERPOLATE_INFERENCE_SIMP = False

    # P2P transfer
    P2P_TRANSFER = True

    # Temp
    INTERPOLATE_INFERENCE_TEMP = False  # synthesize
    INTERPOLATE_INFERENCE_TEMP_EMO_DETECT = False # evaluation emoTempInterp

    # Freq
    INTERPOLATE_INFERENCE_FREQ = False   # synthesize
    INTERPOLATE_INFERENCE_FREQ_PSD_MATCH = False  # evaluation emoFreqInterp

    # inference without interpolation
    if NO_INTERPOLATION:
        # INPUT
        emo_labels = ["Angry", "Sad", "Happy"]
        guidence_strengths = [0.0, 0.5, 1.0, 3.0, 5.0]  # 0: only cond_diff

        for emo_label in emo_labels:
            for guidence_strength in guidence_strengths:
                emo_label_tensor = torch.LongTensor(get_emo_label(emo_label, emo_num_dict)).cuda()
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
                          guidence_strength=guidence_strength,
                          stoc=stoc
                          )

    if P2P_TRANSFER:
        torch.set_printoptions(threshold=10_000)
        # conditions 1 (base style)
        speaker_id1 = 19  # 15
        emo_label1 = "Sad"
        ref1_id = emo_melstyle_dict2[emo_label1].split(".")[0]

        spk_tensor1 = torch.LongTensor([speaker_id1]).cuda() if not isinstance(speaker_id1, type(None)) else None
        emo_label_tensor1 = torch.LongTensor(get_emo_label(emo_label1, emo_num_dict)).cuda()
        melstyle_tensor1 = torch.from_numpy(np.load(
            melstyle_dir + "/" + emo_melstyle_dict2[emo_label1])).cuda()
        melstyle_tensor1 = melstyle_tensor1.unsqueeze(0).transpose(1, 2)

        # conditions 2 (p-level style)
        speaker_id2 = 19  # 15
        emo_label2 = "Angry"
        ref2_id = emo_melstyle_dict2[emo_label2].split(".")[0]

        spk_tensor2 = torch.LongTensor([speaker_id2]).cuda() if not isinstance(speaker_id2, type(None)) else None
        emo_label_tensor2 = torch.LongTensor(get_emo_label(emo_label2, emo_num_dict)).cuda()
        melstyle_tensor2 = torch.from_numpy(np.load(
            melstyle_dir + "/" + emo_melstyle_dict2[emo_label2])).cuda()
        melstyle_tensor2 = melstyle_tensor2.unsqueeze(0).transpose(1, 2)

        #ref_id = "0019_000403"
        # p2p (resource and target) phoneme index
        ## still 7 - 10 (8 9: till) leave (10 11) forest  18 - 26 (rest: 21 25)
        ## ?? get it from x with 148? ??
        tgt_p_start, tgt_p_end = 10, 14  # need check
        ref_p_start, ref_p_end = 7, 9  # till 6 7

        # p2p mode
        p2p_mode = "crossAttn"  # noise or crossAttn
        random_seeds = 2   # v100: slight "still", v102, strong "still"

        """
        from text.phoneme import get_phoneme, get_phonemeID
        # reference phoneme start/end
        print(get_phoneme(ref2_text))
        ref2_transfer_phonme = "?" # <- select from above
        ref_p_start, ref_p_end = get_phonemeID(ref2_text, ref2_transfer_phonme)

        for i, text in enumerate(texts):
            print(get_phoneme(ref2_text))
        for i, phone in enumerate(tgt_phnms):
            tgt_p_start, tgt_p_end = get_phonemeID(ref2_text, phone)
        # target phoneme start/end
        tgt_phnms = ["", "", ""]  # len == len(texts)
        tgt_
        """
        ########### CHECK1: ref/tgt id check ############
        with open(syn_txt, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines()]
        index_list = parse_filelist(ref_indx_dir + "/ref_index.txt")
        ref2_text = [text for basename, speaker, phone, text in index_list if basename == ref2_id][0]
        ref2_phnms = text_to_arpabet(ref2_text, cleaner_names=["english_cleaners"], dictionary=cmu)  # ["p1", "p2"]
        ref2_phnm = ref2_phnms[ref_p_start:ref_p_end]
        print("reference phn[{}:{}]: {} <- {}".format(ref_p_start, ref_p_end, ref2_phnm, ref2_text))
        # tgt
        for i, text in enumerate(texts):
            tgt_phnms = text_to_arpabet(text, cleaner_names=["english_cleaners"], dictionary=cmu)  # ["p1", "p2"]
            tgt_phnm = tgt_phnms[tgt_p_start:tgt_p_end]
            #print(tgt_phnms)
            print("target phn[{}:{}]:{}  <- {}".format(tgt_p_start, tgt_p_end, tgt_phnm, text))  ## h e w i l <- w is 2

        # Output
        out_dir = f"{logs_dir}chpt{chpt_n}_time{time_steps}_spk{speaker_id}_emo{emo_label}"
        if not os.path.isdir(out_dir):
            Path(out_dir).mkdir(exist_ok=True)

        for i in [random_seeds]:
            if True:
                torch.manual_seed(random_seeds)
                inference_p2p_transfer_by_attnMask(
                    configs,
                    model_name,
                    chk_pt,
                    syn_txt=syn_txt,
                    time_steps=time_steps,

                    spk1=spk_tensor1,
                    emo_label1=emo_label_tensor1,
                    melstyle1=melstyle_tensor1,

                    spk2=spk_tensor2,
                    emo_label2=emo_label_tensor2,
                    melstyle2=melstyle_tensor2,

                    out_dir=out_dir,
                    guidence_strength=guidence_strength,
                    stoc=stoc,
                    ref_p_start=ref_p_start,
                    ref_p_end=ref_p_end,
                    tgt_p_start=tgt_p_start,
                    tgt_p_end=tgt_p_end,
                    p2p_mode=p2p_mode,
                    output_pre="p2p_sample_v{}_".format(str(i))
                )
            if True:
                # without p2p transfer
                torch.manual_seed(random_seeds)
                inference(configs,
                          model_name,
                          chk_pt,
                          syn_txt=syn_txt,
                          time_steps=time_steps,
                          spk=spk_tensor1,
                          emo_label=emo_label_tensor1,
                          melstyle=melstyle_tensor1,
                          out_dir=out_dir,
                          guidence_strength=guidence_strength,
                          stoc=stoc,
                          output_pre="no_p2p_sample_v{}_".format(str(i))
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
            guidence_strength=guidence_strength,
            stoc=stoc
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
                   f"{emo_label2}_{mask_range_tag}_timeSplit_{guidence_strength}")
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
            guidence_strength=guidence_strength,
            stoc=stoc
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
            guidence_strength=guidence_strength,
            stoc=stoc
        )

