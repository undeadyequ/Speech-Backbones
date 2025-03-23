import sys, os, yaml, json
import torch
import datetime as dt
import numpy as np
from numba.core.typing.npydecl import Angle
from scipy.io.wavfile import write
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Literal

from scipy.signal import ellip

sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS')
import shutil
from distutils.command.config import config
from GradTTS.read_model import get_model, get_vocoder, read_model_configs, modify_submodel_config
from GradTTS.text import text_to_sequence, cmudict, text_to_arpabet
from GradTTS.text.symbols import symbols

from GradTTS.util.util_config import convert_to_generator_input
from GradTTS.util.util_prosodic_feature import extract_pitch, extract_energy
from GradTTS.util.util_alignment import extract_tgt
from GradTTS.utils import intersperse, get_emo_label
from GradTTS.const_param import emo_melstyleSpk_dict, config_dir, logs_dir, melstyle_dir, wav_dir, wav_dict, emo_num_dict, logs_dir_par
from GradTTS.exp.utils import get_ref_pRange

from GradTTS.exp.visualization import show_pitch_contour, show_energy_contour, compare_pitch_contour, show_attn_map
from GradTTS.model.maskCreation import create_p2p_mask, DiffAttnMask

from GradTTS.preprocessor.preprocessor_v2 import PreprocessorExtract
import time

cmu = cmudict.CMUDict('/home/rosen/Project/Speech-Backbones/GradTTS/resources/cmu_dictionary')

def get_styles_wavs_from_dict(style_dict, emo_type="index", emo_num=5):
    styles = []
    for emo, mel_spk in style_dict.items():
        melstyle, spk = mel_spk
        emo_tensor = get_emo_label(emo, emo_num_dict, emo_type="index")

        spk_tensor = torch.LongTensor([spk]).cuda()
        melstyle_tensor = torch.from_numpy(np.load(
            melstyle_dir + "/" + melstyle)).cuda()
        melstyle_tensor = melstyle_tensor.unsqueeze(0).transpose(1, 2)
        wav_n = os.path.join(wav_dir, melstyle.split(".")[0] + ".wav")
        styles.append((emo, spk, wav_n, emo_tensor, melstyle_tensor, spk_tensor))
    return styles


def get_synText_from_file(synText_f):
    with open(syn_txt, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    return texts


def evaluate_prosody_similarity(
        styles: List[Tuple[Any, Any, Any, Any, Any, Any]],
        synTexts: List[str],
        extra_input_dict: dict,
        model_name1: str,
        model_name2: str,
        chk_pt1: str,
        chk_pt2: str,
        configs1: Tuple[Any, Any, Any],
        configs2: Tuple[Any, Any, Any],
        out_dir: str = "option",
        vis_only: bool = True,
        vis_config: dict = {"row_col_num": (2, 3)},
        fs_model: int = 22050,
        out_png = "result/{}_similarity_{}.png",
        start_step=1,
        end_step=2,
):
    """
    0. Get input and model
        - Style: emotion lable, Reference speech, speaker
            - one style  : tuple(emo, ref, spk)
            - multi sytle:   {emo: [(ref, spk))]}
        - synTexts
            - one text: "t1"
            - multi text: ["t1", "t2"]
        - Generator : name, configs (pre/model/train)
        - includes gradTTS w/o attnMask, w/o noise manipulation
        - Vocoder: name, vocder_config
    1. Synthesize speech and save it/attn in folder
    2. Compute psd and save to json  ->   {"model_1": {"pitch": [[], [], []], "energy": [[], []]}}
    3. Vis psd contour given json file
    4. Vis attention map given attn file

    sub_leg_data_dict: {emo:{model1: [data0, data1]}}

    args:
        styles: [(emo_lab, melstyle, spk), ...]
        out_png: {psd}_similarity_{model}.png

    Returns:
        output_dir
            model1
                emo1_0.wav
                emo1_0.lab
            model1_out
                energy/mel/pitch
                    emo1_0.npy
            model1_tg
                emo1_0.TextGrid
            model1_uAttn
                emo1_0.npy  [(t1, [[h, l_q_phoneme_80, l_k_frame_len], [...]]), (t2)]
    """
    ####### 0. Get input and model  #######
    time_steps = 50
    temperature = 1.5
    stoc = False
    length_scale = 0.91
    guidence_strength = 3.0

    vocoder = get_vocoder()

    ####### 1. Synthesize speech and save it/attn in folder #######
    if start_step <= 1 <= end_step:
        print("1. Synthesize speech and save it/attn in folder")
        with torch.no_grad():
            for model_n, configs, chk_pt in ([model_name1, configs1, chk_pt1], [model_name2, configs2, chk_pt2]):
                # skip gradtts (temp)
                #if model_n == "gradtts":
                #    continue
                generator = get_model(configs, model_n, chk_pt)
                for i, style in enumerate(styles):  # Emo/mels change, spk same
                    emo, spk, wav_n, emo_tensor, melstyle_tensor, spk_tensor = style
                    for j, text in enumerate(synTexts):
                        print(text, model_n)
                        x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
                        x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                        # general input
                        generator_input = convert_to_generator_input(
                            model_n, x, x_lengths, time_steps, temperature, stoc, spk_tensor, length_scale,
                            emo_tensor, melstyle_tensor, guidence_strength
                        )
                        # extra input
                        if extra_input_dict is not None:
                            extra_input = extra_input_dict[model_n]
                            extra_input_dynamic = get_dynamic_extra_input(model_n, extra_input,
                                                                          x, x_lengths, emo_tensor, melstyle_tensor, spk_tensor)
                            generator_input.update(extra_input_dynamic)
                        t = dt.datetime.now()

                        out = generator.forward(**generator_input)
                        if len(out) == 4:
                            y_enc, y_dec, attn, unet_attn = out
                            cross_attn = None                   # [(t1, [[h, l_q_phoneme_80, l_k_frame_len], [...]]), (t2)]
                        else:
                            y_enc, y_dec, attn, unet_attn, cross_attn = out  # cross_attn = [[h, t_t, t_s] * t_sel]

                        # vocoding
                        t = (dt.datetime.now() - t).total_seconds()
                        print(f'Grad-TTS RTF: {t * fs_model / (y_dec.shape[-1] * 256)}')
                        audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

                        # write wav
                        out_sub_dir = os.path.join(out_dir, model_n)
                        if not os.path.isdir(out_sub_dir):
                            Path(out_sub_dir).mkdir(exist_ok=True)
                        write(f'{out_sub_dir}/{emo}_{j}.wav', fs_model, audio)

                        # write text file (for alignment)
                        txt_f = f'{out_sub_dir}/{emo}_{j}.lab'
                        with open(txt_f, "w") as file1:
                            # Writing data to a file
                            file1.write(text)

                        # save unet_attn in dir ("model1": emo1_id1.npy, emo1_id2.npy)
                        out_attn_sub_dir = os.path.join(out_dir, model_n + "_attn")
                        if not os.path.isdir(out_attn_sub_dir):
                            Path(out_attn_sub_dir).mkdir(exist_ok=True)
                        attn_f = f'{out_attn_sub_dir}/{emo}_{i}.npy'
                        if cross_attn is not None:
                            #cross_attn = np.array(cross_attn)         # [(t1, [[h, l_q_phoneme_80, l_k_frame_len], [...]]), (t2)]
                            cross_attn = cross_attn.cpu().numpy()
                            np.save(attn_f, cross_attn)

        # copy reference wav for own text
        out_sub_dir = os.path.join(out_dir, "reference")
        if not os.path.isdir(out_sub_dir):
            print("1.1. Do reference wav copy!")
            Path(out_sub_dir).mkdir(exist_ok=True)
            for i, style in enumerate(styles):
                emo, ref = style[0], style[2]
                shutil.copyfile(ref, out_sub_dir + f"/{emo}_0.wav")
                # write text file (for alignment)
                txt_f = f'{out_sub_dir}/{emo}_0.lab'
                with open(txt_f, "w") as file1:
                    # Writing data to a file
                    file1.write(synTexts[0])

                # copy reference wav for bk syn text
                for j, _ in enumerate(synTexts[1:]):
                    shutil.copyfile(ref, out_sub_dir + f"/{emo}_{j + 1}.wav")
                    shutil.copyfile(txt_f, f'{out_sub_dir}/{emo}_{j + 1}.lab')
        else:
            print("1.1. Skip reference wav copy cause it has been Done!")
    else:
        print("1. Skip speech synthesis cause it has been Done!")


    ####### 2. Compute psd and save to json  #######
    # prosody_info_dict
    prosody_dict_json = os.path.join(out_dir, "prosody_dict.json")  # {"emo1": {"model1": {"psd1/phone": []}}}

    if start_step <= 2 <= end_step:
        print("2. Compute psd and save to json!")
        prosody_dict = dict()
        preprocessor = PreprocessorExtract(configs1[0])
        for model_n in ["reference", model_name1, model_name2]:
            ## Create output dir
            out_sub_dir = os.path.join(out_dir, model_n)    # audio_txt folder
            tg_dir = os.path.join(out_dir, model_n + "_tg") # textgrid folder
            out_psd_dir = os.path.join(out_dir, model_n + "_out") # psd folder
            ## Do MFA
            if not os.path.isdir(tg_dir):
                ## Conduct alignment by "python mfa_dir.py ~/data/test_data/ english_mfa ~/data/tg/" under aligner conda enviroment
                print("####################")
                print("Please conduct alignment by 'python mfa_dir.py {} english_mfa {}' under aligner conda enviroment".format(
                    out_sub_dir, tg_dir
                ))
                print(time.sleep(5))
                print("####################")
                sys.exit()
            # Extract PSD given wav/tgt and save json
            speech_list = [speech for speech in os.listdir(out_sub_dir) if speech.endswith(".wav")]
            for speech in speech_list:
                speech_f = os.path.join(out_sub_dir, speech)
                emo_id = speech.split("_")[0]
                if emo_id not in prosody_dict.keys():
                    prosody_dict[emo_id] = dict()
                if model_n not in prosody_dict[emo_id].keys():
                    prosody_dict[emo_id][model_n] = {
                        "phonemes": [],
                        "pitch": [],
                        "energy": [],
                        "duration": []
                    }
                tg_path = os.path.join(
                    tg_dir, "{}.TextGrid".format(os.path.basename(speech_f).split(".")[0]))
                phonemes, pitch, energy, mels, duration = preprocessor.extract_pitch_energy_mel(speech_f,
                                                                            tg_path=tg_path,
                                                                            out_dir=out_psd_dir,
                                                                            average_phoneme=True,
                                                                            save_npy=True)
                prosody_dict[emo_id][model_n]["phonemes"].append(phonemes.split(" "))
                prosody_dict[emo_id][model_n]["pitch"].append(pitch.tolist())
                prosody_dict[emo_id][model_n]["energy"].append(energy.tolist())
                prosody_dict[emo_id][model_n]["duration"].append(duration)

        with open(prosody_dict_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(prosody_dict, sort_keys=True, indent=4))
    else:
        with open(prosody_dict_json, "r") as f:
            prosody_dict = json.load(f)
        print("2. Skip prosody extraction cause it has been done!")


    ####### 3. Vis psd contour given json file  #######
    pitch_dict_for_vis = dict()  # for_vis: {"emo1": {"model1": list(psd_len)}}}
    if start_step <= 3 <= end_step:
        print("3. Vis psd contour given json file!")
        with open(prosody_dict_json, "r") as f:
            prosody_dict = json.load(f)
        rc_num = (3, 2)
        legend_mark = ("*", "v", ".")
        xy_label_sub = ("frame", "Hz")

        for vis_speechid in [0, 1]:
            #vis_speechid = 1   # 1: parallel data, 0~: Non-parallel data
            phonems = prosody_dict["Angry"]["reference"]["phonemes"][vis_speechid]

            if vis_speechid == 0:
                show_txt = True
                title_extra = "Non_parallel task"
            else:
                show_txt = False
                title_extra = "Parallel task"

            print("3.1. do pitch comaration visualization on {}".format(" ".join(
                prosody_dict["Angry"]["reference"]["phonemes"][vis_speechid])))

            for emo, m_psd in prosody_dict.items():
                pitch_dict_for_vis[emo] = {}
                for model, psd in m_psd.items():
                    if model not in pitch_dict_for_vis[emo].keys():
                        pitch_dict_for_vis[emo][model] = []
                    if model != "reference":
                        pitch_dict_for_vis[emo][model] = psd["pitch"][vis_speechid]
                    else:
                        pitch_dict_for_vis[emo][model] = psd["pitch"][0]
            compare_pitch_contour(
                pitch_dict_for_vis,
                rc_num,
                legend_mark=legend_mark,
                xy_label_sub=xy_label_sub,
                xtickslab=phonems,
                out_png=out_png.format("pitch", title_extra.split(" ")[0]),
                title="Pitch contour of speech synthesized on reference, emoMix, proposed ({})".format(title_extra),
                show_txt=show_txt)

            print("5. do energy comaration visualization on {}st txt".format(vis_speechid))
            rc_num = (3, 2)
            legend_mark = ("*", "v", ".")
            xy_label_sub = ("frame", "Hz")
            energy_dict_for_vis = dict()   # {"emo": {"model1": []}}
            for emo, m_psd in prosody_dict.items():
                energy_dict_for_vis[emo] = {}
                for model, psd in m_psd.items():
                    if model not in energy_dict_for_vis[emo].keys():
                        energy_dict_for_vis[emo][model] = []
                    if model != "reference":
                        energy_dict_for_vis[emo][model] = psd["energy"][vis_speechid]
                    else:
                        energy_dict_for_vis[emo][model] = psd["energy"][0]
            compare_pitch_contour(
                energy_dict_for_vis,
                rc_num,
                legend_mark=legend_mark,
                xy_label_sub=xy_label_sub,
                xtickslab=phonems,
                out_png=out_png.format("energy", title_extra.split(" ")[0]),
                title="Energy contour of speech synthesized on reference, emoMix, proposed ({})".format(title_extra),
                show_txt=show_txt
            )
    else:
        print("Skip Vis psd contour!")

    ####### 4. Vis attention map given attn file #######
    # attn_info_dict (all_in_one) and attn_vis_dict
    attn_map_dict = dict()                  # {"model1": {"emo1": [np.array((t, h, p1_len, p2_len])), [] ]}}}  # first, last
    attn_map_for_vis = dict()               # {"model1": {"emo1": np.array([p1_len, p2_len])}}}  <- choose t and h from attn_map_dict

    t1, b1, h1 = 0, 1, 0

    if start_step <= 4 <= end_step:
        print("4. Vis attention map given attn file")
        tgt_ref_dir = os.path.join(out_dir, "reference_tg")
        for vis_speechid in [0, 1]:
            # 1: parallel data, 0~: Non-parallel data
            syn_phonems = prosody_dict["Angry"][model_name1]["phonemes"][vis_speechid]
            ref_phonemes = prosody_dict["Angry"]["reference"]["phonemes"][1]

            for model_n in [model_name1, model_name2]:
                out_attn_sub_dir = os.path.join(out_dir, model_n + "_attn")
                tgt_syn_dir = os.path.join(out_dir, model_n + "_tg")
                attn_map_dict[model_n] = {}
                attn_map_for_vis[model_n] = {}
                
                for attn_map_n in os.listdir(out_attn_sub_dir):
                    emo, id = attn_map_n.split("_")
                    # get dur, phone from syn/ref tgt
                    tgt_name = attn_map_n.split(".")[0] + ".TextGrid"
                    tgt_ref_f = os.path.join(tgt_ref_dir, tgt_name)
                    tgt_syn_f = os.path.join(tgt_syn_dir, tgt_name)
                    #ref_phonemes, ref_duration, _, _ = extract_tgt(tgt_ref_f)
                    #syn_phonemes, syn_duration, _, _ = extract_tgt(tgt_syn_f)

                    # get p2f_score from saved file
                    attn_map_f = os.path.join(out_attn_sub_dir, attn_map_n)
                    cross_attn = np.load(attn_map_f, allow_pickle=True)  # [t_sel, layers. h, b, t_t, t_s]

                    attn_map_for_vis[model_n][emo] = cross_attn[t1, b1, 0, h1]  # o tis batch

                    # compute p2p_score from p2f_score and dur, phone
                    """
                    p2f_score = np.load(attn_map_f, allow_pickle=True)  # [(t1, [[h, l_q_phoneme_80, l_k_frame_len], [...]])]
                    for t, (p2f_first, p2f_last) in p2f_score:
                        p2p_first = get_p2p_from_p2f_score(p2f_first,
                                                            ref_duration,
                                                            ref_phonemes,
                                                            syn_phonemes)  # (h, p1_len, p2_len)                       
                        p2p_last = get_p2p_from_p2f_score(p2f_last,
                                    ref_duration,
                                    ref_phonemes,
                                    syn_phonemes)  # (h, p1_len, p2_len)
                        attn_map_dict[model_n][emo] = [(t, p2p_first), (t, p2p_last)]
                        if t == vis_t:
                            attn_map_for_vis[model_n][emo] = (p2p_first[vis_h, :, :], p2p_last[vis_h, :, :])
                    """

        # vis attention given p2p_score, phone of each model
        for model_n, emo_attn_dict in attn_map_for_vis.items():  # emo_attn: {"emo": (h, p1_len, p2_len)}
            rc_num = (3, 2)
            xy_label_sub = ("phoneme", "phoneme")

            title_extra = "non_parallel"
            show_attn_map(
                emo_attn_dict,
                rc_num,
                xy_label_sub=xy_label_sub,
                xtickslab=ref_phonemes,
                ytickslab=syn_phonems,
                out_png=out_png.format(model_n, "_t_{}_b_{}_h_{}".format(t1, b1, h1)),
                # "result/{}_similarity_{}.png"
                title="Attention map in crossAttention after masking ({})".format(title_extra))

            """
            attn_map_for_vis_first = [{emo: first_last[0]} for emo, first_last in emo_first_last_attn.items()]
            attn_map_for_vis_last = [{emo: first_last[1]} for emo, first_last in emo_first_last_attn.items()]
            
            show_attn_map(
                attn_map_for_vis_first,
                rc_num,
                xy_label_sub=xy_label_sub,
                xtickslab=ref_phonemes,
                ytickslab=syn_phonems,
                out_png=out_png.format("attn_first", model + "_" + "t_{}_h_{}".format(vis_t, vis_h)),  # "result/{}_similarity_{}.png"
                title="Energy contour of speech synthesized on reference, emoMix, proposed ({})".format(title_extra))
            show_attn_map(
                attn_map_for_vis_last,
                rc_num,
                xy_label_sub=xy_label_sub,
                xtickslab=ref_phonemes,
                ytickslab=syn_phonems,
                out_png=out_png.format("attn_last", "_" + "t_{}_h_{}".format(vis_t, vis_h)),
                title="Energy contour of speech synthesized on reference, emoMix, proposed ({})".format(title_extra))
            """

            
    else:
        print("Skip Vis attention map!")

def get_p2p_from_p2f_score(p2f_score, ref_duration, ref_phonemes, syn_phonemes, f2p_alg="average"):

    """
    Args:
        p2f_score: (h, l_q * d_q, l_k)
        p_dur: (p_len,)
        ref_phonemes: (l_k_phoneme_len,)
        txt_phonemes: (l_q_phoneme_len, )

    Returns:
        p2p_score: (h, l_q_phoneme, l_k_phoneme
    """

    # check size
    h, ld_q, l_k = p2f_score.size()
    p2p_score = np.zeros(h, ld_q / 80, len(ref_phonemes))
    
    frame_ind_start = 0

    # Average on 1st dim
    for i, dur in enumerate(ref_duration):
        frame_index = convert_dur2frameIdx(dur)
        frame_ind_end = frame_ind_start + frame_index
        p2p_score[:, :, i:i + 1] = np.mean(p2f_score[:, :, frame_ind_start:frame_ind_end], dim=2)
        frame_ind_start = frame_ind_end

    # Average on 2nd dim
    for pIndx in range(ld_q / 80):
        p2p_score[:, pIndx:pIndx + 1, :] = np.mean(p2f_score[:, pIndx:pIndx + 80, :], dim=1)
    return p2p_score

def convert_dur2frameIdx(dur, frame_len=128, hop_len=128):
    frameIdx = dur / hop_len + frame_len / 2
    return frameIdx


def get_p2p_from_p2f_score(p2p_score, p_dur, phonemes):
    """

    Args:
        p2p_score: (b, h, l_q * d_q, l_k)
        p_dur: (p_len,)
        phonemes: (p_len,)

    Returns:

    """
    pass


def get_dynamic_extra_input(model_n, extra_input, x, x_lengths, emo_lab, melstyle, spk):
    if model_n == "gradtts_cross":
        tgt_pRange = extra_input["tgt_pRange"]  # phoneme range
        ref_pRange = extra_input["ref_pRange"]  # phoneme range

        #ref_pFrame_range = get_ref_pRange(ref_pRange[0], ref_pRange[1], ref2_id)

        ref_pFrame_range = (ref_pRange[0] * 5, ref_pRange[1] * 5)  # tempt use

        p2p_mask1 = create_p2p_mask(
            tgt_size=x.shape[1],
            ref_size=melstyle.shape[2],
            dims=4,
            tgt_range=tgt_pRange,
            ref_range=ref_pFrame_range
        )
        p2p_mask1 = p2p_mask1.cuda()
        p2p_mask2 = 1 - p2p_mask1

        extra_input_dynamic = {
            "attn_hardMask": p2p_mask2
        }
    else:
        extra_input_dynamic = {
            "attn_hardMask": None
        }
    return extra_input_dynamic

def get_extra_input(model_n, extract_input_require):
    if model_n == "gradtts_cross":
        extra_input = get_extra_input_for_gCross(
            **extract_input_require
        )
    else:
        extra_input = None
    return extra_input


def get_extra_input_for_gCross(
        model_n,
        x,
        melstyle,
        tgt_pRange,
        ref_pFrame_range
):
    """
    get model-specific input
    Args:
        model_n:
        x:
        melstyle:
        tgt_pRange:
        ref_pFrame_range:

    Returns:

    """
    if model_n == "gradtts_cross":
        p2p_mask1 = create_p2p_mask(
            tgt_size=x.shape[1],
            ref_size=melstyle.shape[2],
            dims=4,
            tgt_range=tgt_pRange,
            ref_range=ref_pFrame_range
        )
        p2p_mask1 = p2p_mask1.cuda()
        p2p_mask2 = 1 - p2p_mask1

        extra_input = {
            "attn_hardMask": p2p_mask2
        }
    else:
        extra_input = None

    return extra_input


if __name__ == "__main__":
    # input
    syn_txt = "../resources/filelists/synthesis1_closetext_onlyOne.txt"
    # emo, spk, wav_n, emo_tensor, melstyle_tensor, spk_tensor
    styles = get_styles_wavs_from_dict(emo_melstyleSpk_dict)  # emotion changed

    # output
    out_dir = f"{logs_dir}evalobj_prosody_sim"
    out_res_dir = out_dir + "/result_stditCompare"
    out_png = out_res_dir + "/{}_similarity_{}.png"  #

    # extra input (Only for p2p style transfer)
    """
    tgt_pRange = (8, 9)
    ref_pRange = (8, 9)
    extra_input_dict = {
        "gradtts_cross": {
            "tgt_pRange": (8, 9),
            "ref_pRange": (8, 9)
        },
        "gradtts": None,
    }
    """

    # set model
    model_name1 = "STDitHard" # "gradtts"  #
    model_name2 = "STDitLossGuide" # "gradtts_cross" #
    model_name3 = "STDitLossGuideVQ"
    vocoder_name = "hifigan"

    # set model config
    model_config_dict = {
        "gradtts": ("preprocess_gradTTS.yaml", "model_gradTTS_v3.yaml", "train_gradTTS.yaml"),
        "gradtts_cross": ("preprocess_gradTTS.yaml", "model_gradTTS_v3.yaml", "train_gradTTS.yaml"),
        "STDit": ("preprocess_styleAlignedTTS.yaml", "model_styleAlignedTTS.yaml", "train_styleAlignedTTS.yaml"),
    }
    submodel_modify_dict = {
        "STDitMocha": [
            {"path": {"log_dir": "/home/rosen/Project/Speech-Backbones/GradTTS/logs/styleEnhancedTTS_stditMocha_norm_hardMonMask/"}},  # train.yaml
            {"stditMocha": {"monotonic_approach": "mocha"}, "ref_encoder": "mlp", "ref_embedder": "wav2vect2"}],                                                                          # model.yaml
        "STDitHard": [
            {"path": {"log_dir": "/home/rosen/Project/Speech-Backbones/GradTTS/logs/styleEnhancedTTS_stditMocha_norm_hardMonMask/"}},
            {"stditMocha": {"monotonic_approach": "hard"}}],
        "STDitLossGuide": [
            {"path": {"log_dir": "/home/rosen/Project/Speech-Backbones/GradTTS/logs/styleEnhancedTTS_stditMocha_norm_guidloss/"}},
            {"loss": {"guided_attn": True}, "stditMocha": {"monotonic_approach": "guidloss"}, "ref_encoder": "mlp", "ref_embedder": "wav2vect2"}],
        "STDitLossGuideVQ": [
            {"path": {
                "log_dir": "/home/rosen/Project/Speech-Backbones/GradTTS/logs/styleEnhancedTTS_stditMocha_norm_guideloss_vqvae/"}},
            {"loss": {"guided_attn": True}, "stditMocha": {"monotonic_approach": "guidloss"}, "ref_encoder": "vaeEma",
             "ref_embedder": "mel"}],

    }

    chk_pt1 = submodel_modify_dict[model_name1][0]["path"]["log_dir"] + "models/grad_145.pt"  # "models/grad_400.pt"
    chk_pt2 = submodel_modify_dict[model_name2][0]["path"]["log_dir"] + "models/grad_147.pt"  # "models/grad_400.pt"

    pmt_configs1 = read_model_configs(config_dir, model_config_dict["STDit"])  # ******** Moidfy STDit if GradTTS +++++++++
    pmt_configs2 = read_model_configs(config_dir, model_config_dict["STDit"])

    # modify model parameter for modelsub
    pmt_configs1 = modify_submodel_config(pmt_configs1,
                                          train_mody=submodel_modify_dict[model_name1][0],
                                          model_mody=submodel_modify_dict[model_name1][1])
    pmt_configs2 = modify_submodel_config(pmt_configs2,
                                          train_mody=submodel_modify_dict[model_name2][0],
                                          model_mody=submodel_modify_dict[model_name2][1])

    synTexts = get_synText_from_file(syn_txt)
    if not os.path.isdir(out_res_dir):
        Path(out_res_dir).mkdir(exist_ok=True, parents=True)

    # vis config
    pitch_vis_config = {
        "row_col_num": (2, 3),
        "title": "Pitch similarity comparation",
        "subtitle": ["Angry", "Happy"],
        "pair_legend": ("GradTTS", "InterpTTS"),
        "pair_mark": ("*", "o"),
        "xy_label_sub": ("frame", "pitch")
    }

    energy_vis_config = {
        "row_col_num": (2, 3),
        "title": "Energy similarity comparation",
        "subtitle": ["Angry", "Happy"],
        "pair_legend": ("GradTTS", "InterpTTS"),
        "pair_mark": ("*", "o"),
        "xy_label_sub": ("frame", "energy")}

    evaluate_prosody_similarity(
        styles=styles,
        synTexts=synTexts,
        extra_input_dict=None,
        model_name1=model_name1,
        model_name2=model_name2,
        chk_pt1=chk_pt1,
        chk_pt2=chk_pt2,
        configs1=pmt_configs1,
        configs2=pmt_configs2,
        out_dir=out_dir,
        vis_only=True,
        vis_config=pitch_vis_config,
        fs_model=22050,
        out_png=out_png,
        start_step=1,
        end_step=4,
    )