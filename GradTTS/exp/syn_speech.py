import sys, os, yaml, json
from itertools import accumulate

import torch
import datetime as dt
import numpy as np
from scipy.io.wavfile import write
from pathlib import Path
sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS')
from GradTTS.read_model import get_model, get_vocoder, get_model1
from GradTTS.text import text_to_sequence, cmudict, text_to_arpabet, text2id_iterp, text2phone_iterp
from GradTTS.text.symbols import symbols

from GradTTS.util.util_config import convert_to_generator_input
from GradTTS.utils import intersperse, index_nointersperse

cmu = cmudict.CMUDict('/home/rosen/Project/Speech-Backbones/GradTTS/resources/cmu_dictionary')

text2id = {
    "he was still in the forest!": 0,
    "and we are so thirsty!": 1,
    "i do not eat bread.": 2,
    "let go of my top knot.": 3,
    "and they did push so!": 4,
    "he was killed by an arrow.": 5,
    "it looks much better.": 6,
    "kitty, can you play chess?": 7,
    "however, somebody killed something.": 8,
    "it would be a hard choice.": 9
}


def syn_speech_from_models(model1_set, model2_set, styles, synTexts, out_dir, fs_model=16000):
    """
    1. Synthesize speech by given models_set (model_name, configs, chk_pt)
    2. save wav, txt, attention files
    Args:
        model1_set:
        model2_set:
        styles: [[emo, spk, wav_n, emo_tensor, melstyle_tensor, spk_tensor], []]
        synTexts:
        out_dir:
    Returns:
    """

    model_name1, model_config1, chk_pt1 = model1_set
    model_name2, model_config2, chk_pt2 = model2_set
    vocoder = get_vocoder()

    attn_dict = {}   # for record qk_dur and phoneme for crossAttn visualization

    with torch.no_grad():
        for model_n, configs, chk_pt in ([model_name1, model_config1, chk_pt1], [model_name2, model_config2, chk_pt2]):
            # skip model_n
            #if model_n == "stditCross_noguide_codec":
            #    continue
            ########## Setup Generator ##########
            preprocess_config, model_config, train_config = configs
            bone_n, fineAdjNet, input_n = model_n.split("_")
            chk_pt = train_config["path"]["log_dir"] + "models/" + chk_pt

            add_blank = preprocess_config["feature"]["add_blank"]
            nsymbols = len(symbols) + 1 if add_blank else len(symbols)
            generator = get_model1(model_config, bone_n, chk_pt, nsymbols=nsymbols)

            inference_configs = model_config["inference"]
            stoc, time_steps, temperature, length_scale = \
                inference_configs["stoc"], inference_configs["n_timesteps"], \
                    inference_configs["temperature"], inference_configs["length_scale"],

            for i, (style, text) in enumerate(zip(styles[-5:], synTexts[-5:])):
                txtid = text2id[text]
                ########## Synthesize ##########
                if len(style) == 7:
                    emo, spk, wav_n, txt, emo_tensor, melstyle_tensor, spk_tensor = style
                    syl_start = None
                else:
                    emo, spk, wav_n, txt, emo_tensor, melstyle_tensor, spk_tensor, syl_start = style
                if fineAdjNet != "guideSyl":
                    syl_start = None
                print("synthesize {} by {}".format(text, model_n))
                x = torch.LongTensor(text2id_iterp(text)).cuda()[None]
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                melstyle_tensor_lengths = torch.LongTensor([melstyle_tensor.shape[-1]]).cuda()
                phonemes = text2phone_iterp(text)

                #syllable_indexs = text_to_syllable(text, dictionary=cmu)
                # Create generator input
                generator_input = convert_to_generator_input(
                    model_n, x, x_lengths, time_steps, temperature, stoc, spk_tensor, length_scale,
                    emo_tensor, melstyle_tensor, melstyle_tensor_lengths, syl_start=syl_start)

                ########## Start synthesis ##########
                t = dt.datetime.now()
                out = generator.forward(**generator_input)
                if len(out) == 4:
                    y_enc, y_dec, attn, unet_attn = out
                    cross_attn = None  # [(t1, [[h, l_q_phoneme_80, l_k_frame_len], [...]]), (t2)]
                else:
                    y_enc, y_dec, attn, unet_attn, cross_attn, q_dur, k_dur = out  # cross_attn = [[h, t_t, t_s] * t_sel]

                if len(phonemes) != len(q_dur.squeeze(0)) or len(phonemes)!= len(k_dur.squeeze(0)):
                    print("Error! x, phone, q_dur, k_dur len should same: {} {} {} {}".format(len(x.squeeze(0)), len(phonemes), len(q_dur.squeeze(0)), len(k_dur.squeeze(0))))

                # Vocoding
                t = (dt.datetime.now() - t).total_seconds()
                #print(f'Grad-TTS RTF: {t * fs_model / (y_dec.shape[-1] * 256)}')
                audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

                ########## Write result ##########
                out_speech_dir = os.path.join(out_dir, model_n)
                out_crossAttn_dir = os.path.join(out_dir, model_n + "_attn")
                speech_id = f'spk{spk}_{emo}_txt{txtid}'

                # Write wav in out_sub_dir
                if not os.path.isdir(out_speech_dir):
                    Path(out_speech_dir).mkdir(exist_ok=True)
                write(f'{out_speech_dir}/{speech_id}.wav', fs_model, audio)

                # Write text file (for alignment)
                txt_f = f'{out_speech_dir}/{speech_id}.lab'
                with open(txt_f, "w") as file1:
                    file1.write(text)

                # Save crossAttn in dir ("model1": emo1_id1.npy, emo1_id2.npy)
                if not os.path.isdir(out_crossAttn_dir):
                    Path(out_crossAttn_dir).mkdir(exist_ok=True)
                attn_f = f'{out_crossAttn_dir}/{speech_id}.npy'
                if cross_attn is not None:
                    # cross_attn = np.array(cross_attn)         # [(t1, [[h, l_q_phoneme_80, l_k_frame_len], [...]]), (t2)]
                    cross_attn = cross_attn.cpu().numpy()
                    np.save(attn_f, cross_attn)

                # recorder
                if emo not in attn_dict.keys():
                    attn_dict[emo] = dict()
                if model_n not in attn_dict[emo].keys():
                    attn_dict[emo][model_n] = {
                        "speechid": [],
                        "phonemes": [],
                        "q_dur": [],
                        "k_dur": []
                    }
                attn_dict[emo][model_n]["phonemes"].append(phonemes)
                attn_dict[emo][model_n]["q_dur"].append(q_dur.squeeze(0).cpu().numpy().tolist())
                attn_dict[emo][model_n]["k_dur"].append(k_dur.squeeze(0).cpu().numpy().tolist())
                attn_dict[emo][model_n]["speechid"].append(speech_id)
    return attn_dict


def syn_speech_from_model_enhance(model1_set, styles, synTexts, out_dir, fs_model=22050,
                                  enh_ind_syn_ref = (2, 2)):
    """
    1. Synthesize speech by given models_set (model_name, configs, chk_pt)
    2. save wav, txt, attention files
    Args:
        model1_set:
        model2_set:
        styles: [[emo, spk, wav_n, emo_tensor, melstyle_tensor, spk_tensor], []]
        synTexts:
        out_dir:
    Returns:
    """

    model_n, configs, chk_pt = model1_set
    vocoder = get_vocoder()

    attn_dict = {}   # for record qk_dur and phoneme for crossAttn visualization

    with torch.no_grad():
        for enh_ind in [None, enh_ind_syn_ref]:
            # skip model_n
            #if model_n == "stditCross_noguide_codec":
            #    continue
            ########## Setup Generator ##########
            preprocess_config, model_config, train_config = configs
            bone_n, fineAdjNet, input_n = model_n.split("_")
            chk_pt = train_config["path"]["log_dir"] + "models/" + chk_pt

            add_blank = preprocess_config["feature"]["add_blank"]
            nsymbols = len(symbols) + 1 if add_blank else len(symbols)
            generator = get_model1(model_config, bone_n, chk_pt, nsymbols=nsymbols)

            inference_configs = model_config["inference"]
            stoc, time_steps, temperature, length_scale = \
                inference_configs["stoc"], inference_configs["n_timesteps"], \
                    inference_configs["temperature"], inference_configs["length_scale"],

            for i, (style, text) in enumerate(zip(styles[-5:], synTexts[-5:])):
                txtid = text2id[text]
                ########## Synthesize ##########
                if len(style) == 7:
                    emo, spk, wav_n, txt, emo_tensor, melstyle_tensor, spk_tensor = style
                    syl_start = None
                else:
                    emo, spk, wav_n, txt, emo_tensor, melstyle_tensor, spk_tensor, syl_start = style
                if fineAdjNet != "guideSyl":
                    syl_start = None
                print("synthesize {} by {}".format(text, model_n))
                x = torch.LongTensor(text2id_iterp(text)).cuda()[None]
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                melstyle_tensor_lengths = torch.LongTensor([melstyle_tensor.shape[-1]]).cuda()
                phonemes = text2phone_iterp(text)

                #syllable_indexs = text_to_syllable(text, dictionary=cmu)
                # Create generator input
                generator_input = convert_to_generator_input(
                    model_n, x, x_lengths, time_steps, temperature, stoc, spk_tensor, length_scale,
                    emo_tensor, melstyle_tensor, melstyle_tensor_lengths, syl_start=syl_start, enh_ind=enh_ind)

                ########## Start synthesis ##########
                t = dt.datetime.now()
                out = generator.forward(**generator_input)
                if len(out) == 4:
                    y_enc, y_dec, attn, unet_attn = out
                    cross_attn = None  # [(t1, [[h, l_q_phoneme_80, l_k_frame_len], [...]]), (t2)]
                else:
                    y_enc, y_dec, attn, unet_attn, cross_attn, q_dur, k_dur = out  # cross_attn = [[h, t_t, t_s] * t_sel]

                if len(phonemes) != len(q_dur.squeeze(0)) or len(phonemes)!= len(k_dur.squeeze(0)):
                    print("Error! x, phone, q_dur, k_dur len should same: {} {} {} {}".format(len(x.squeeze(0)), len(phonemes), len(q_dur.squeeze(0)), len(k_dur.squeeze(0))))

                # Vocoding
                t = (dt.datetime.now() - t).total_seconds()
                #print(f'Grad-TTS RTF: {t * fs_model / (y_dec.shape[-1] * 256)}')
                audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

                ########## Write result ##########
                # output
                enh_ind_name = str(enh_ind[0]) + "_" + str(enh_ind[0]) if enh_ind is not None else None
                model_enh_n = model_n + "_" + enh_ind_name
                out_speech_dir = os.path.join(out_dir, model_enh_n)
                out_crossAttn_dir = os.path.join(out_dir, model_enh_n + "_attn")
                speech_id = f'spk{spk}_{emo}_txt{txtid}'

                # Write wav in out_sub_dir
                if not os.path.isdir(out_speech_dir):
                    Path(out_speech_dir).mkdir(exist_ok=True)
                write(f'{out_speech_dir}/{speech_id}.wav', fs_model, audio)

                # Write text file (for alignment)
                txt_f = f'{out_speech_dir}/{speech_id}.lab'
                with open(txt_f, "w") as file1:
                    file1.write(text)

                # Save crossAttn in dir ("model1": emo1_id1.npy, emo1_id2.npy)
                if not os.path.isdir(out_crossAttn_dir):
                    Path(out_crossAttn_dir).mkdir(exist_ok=True)
                attn_f = f'{out_crossAttn_dir}/{speech_id}.npy'
                if cross_attn is not None:
                    # cross_attn = np.array(cross_attn)         # [(t1, [[h, l_q_phoneme_80, l_k_frame_len], [...]]), (t2)]
                    cross_attn = cross_attn.cpu().numpy()
                    np.save(attn_f, cross_attn)

                # recorder
                if emo not in attn_dict.keys():
                    attn_dict[emo] = dict()
                if model_n not in attn_dict[emo].keys():
                    attn_dict[emo][model_n] = {
                        "speechid": [],
                        "phonemes": [],
                        "q_dur": [],
                        "k_dur": []
                    }
                attn_dict[emo][model_n]["phonemes"].append(phonemes)
                attn_dict[emo][model_n]["q_dur"].append(q_dur.squeeze(0).cpu().numpy().tolist())
                attn_dict[emo][model_n]["k_dur"].append(k_dur.squeeze(0).cpu().numpy().tolist())
                attn_dict[emo][model_n]["speechid"].append(speech_id)
    return attn_dict
