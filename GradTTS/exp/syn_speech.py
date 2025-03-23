import sys, os, yaml, json
import torch
import datetime as dt
import numpy as np
from scipy.io.wavfile import write
from pathlib import Path
sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS')
from GradTTS.read_model import get_model, get_vocoder, get_model1
from GradTTS.text import text_to_sequence, cmudict, text_to_arpabet
from GradTTS.text.symbols import symbols

from GradTTS.util.util_config import convert_to_generator_input
from GradTTS.utils import intersperse

cmu = cmudict.CMUDict('/home/rosen/Project/Speech-Backbones/GradTTS/resources/cmu_dictionary')

def syn_speech_from_models(model1_set, model2_set, styles, synTexts, out_dir, fs_model=22050):
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

    with torch.no_grad():
        for model_n, configs, chk_pt in ([model_name1, model_config1, chk_pt1], [model_name2, model_config2, chk_pt2]):
            ########## Setup Generator ##########
            preprocess_config, model_config, train_config = configs
            bone_n, config_n, input_n = model_n.split("_")
            chk_pt = train_config["path"]["log_dir"] + "/models/" + chk_pt

            add_blank = preprocess_config["feature"]["add_blank"]
            nsymbols = len(symbols) + 1 if add_blank else len(symbols)
            generator = get_model1(model_config, bone_n, chk_pt, nsymbols=nsymbols)

            inference_configs = model_config["inference"]
            stoc, time_steps, temperature, length_scale = \
                inference_configs["stoc"], inference_configs["n_timesteps"], \
                    inference_configs["temperature"], inference_configs["length_scale"],

            for i, (style, text) in enumerate(zip(styles, synTexts)):
                ########## Synthesize ##########
                emo, spk, wav_n, txt, emo_tensor, melstyle_tensor, spk_tensor = style
                print("synthesize:", text, model_n)
                x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()

                # Create generator input
                generator_input = convert_to_generator_input(
                    model_n, x, x_lengths, time_steps, temperature, stoc, spk_tensor, length_scale,
                    emo_tensor, melstyle_tensor)

                t = dt.datetime.now()
                out = generator.forward(**generator_input)
                if len(out) == 4:
                    y_enc, y_dec, attn, unet_attn = out
                    cross_attn = None  # [(t1, [[h, l_q_phoneme_80, l_k_frame_len], [...]]), (t2)]
                else:
                    y_enc, y_dec, attn, unet_attn, cross_attn = out  # cross_attn = [[h, t_t, t_s] * t_sel]

                # Vocoding
                t = (dt.datetime.now() - t).total_seconds()
                print(f'Grad-TTS RTF: {t * fs_model / (y_dec.shape[-1] * 256)}')
                audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

                ########## Write result ##########
                out_speech_dir = os.path.join(out_dir, model_n)
                out_crossAttn_dir = os.path.join(out_dir, model_n + "_attn")
                speech_id = f'spk{spk}_{emo}_txt{i}'

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