import sys, os, yaml, json
import torch
import datetime as dt
import numpy as np
from numba.core.typing.npydecl import Angle
from scipy.io.wavfile import write
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Literal

sys.path.append('/home/rosen/Project/Speech-Backbones')
sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS')
sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS/hifi-gan/')

from GradTTS.exp.utils import convert_vis_psd_json, convert_attn_json, copy_ref_speech, fine_adjust_configs, get_synStyle_from_file, get_synText_from_file
from GradTTS.exp.extract_psd import extract_psd
from GradTTS.exp.syn_speech import syn_speech_from_models, text2id, syn_speech_from_model_enhance
from GradTTS.exp.visualization import vis_psd, vis_crossAttn, vis_psd_enh
from GradTTS.exp.statcz_psd import statcz_psd_mcd
"""
meta_data
vis_data:
    subplot_n -> Approaches -> evaluateData
"""
def main(
        styles: List[Tuple[Any, Any, Any, Any, Any, Any]],
        synTexts: List[str],
        model_name1: str,
        model_name2: str,
        chk_pt1: str,
        chk_pt2: str,
        configs1: Tuple[Any, Any, Any],
        configs2: Tuple[Any, Any, Any],
        out_dir: str = "option",  # middle result
        vis_only: bool = True,
        vis_config: dict = {"row_col_num": (2, 3)},
        fs_model: int = 22050,
        start_step=1,
        end_step=2,
):
    # set output folder and out_png
    bone_n1, fineAdjNet_n1, input_n1 = model_name1.split("_")
    bone_n2, fineAdjNet_n2, input_n2 = model_name2.split("_")
    if bone_n1 == bone_n2:
        out_res_dir = out_dir + "/compare_{}_{}_VS_{}".format(bone_n1, fineAdjNet_n1, fineAdjNet_n2)   # Comparation res image/json/
        model1_n, model2_n = fineAdjNet_n1, fineAdjNet_n2
    else:
        out_res_dir = out_dir + "/compare_{}_VS_{}".format(bone_n1, bone_n2)
        model1_n, model2_n = bone_n1, bone_n2

    out_png = out_res_dir + "/{}_similarity_{}.png"
    if not os.path.isdir(out_res_dir):
        Path(out_res_dir).mkdir(exist_ok=True, parents=True)

    ####### 1. Synthesize speech and save attn file #######
    attn_dict_json = os.path.join(out_res_dir,
                                     "attn_{}_VS_{}.json".format(model1_n, model2_n))  # {"emo1": {"model1": {"speechid/qkdur/phone0": [[], []]}}}
    if start_step <= 1 <= end_step:
        attn_dict = syn_speech_from_models([model_name1, configs1, chk_pt1],
                               [model_name2, configs2, chk_pt2],
                               styles, synTexts, out_dir, fs_model=fs_model)
        with open(attn_dict_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(attn_dict, sort_keys=True, indent=4))

        out_ref_dir = os.path.join(out_dir, "reference")
        if not os.path.isdir(out_ref_dir):
            print("1.1. Do reference wav copy!")
            Path(out_ref_dir).mkdir(exist_ok=True)
            src_wavs = [style[2] for style in styles]
            src_txts = [style[3] for style in styles]
            dst_wavs = [f'{out_ref_dir}/spk{style[1]}_{style[0]}_txt{text2id[style[3]]}.wav' for i, style in enumerate(styles)]
            dst_txts = [f'{out_ref_dir}/spk{style[1]}_{style[0]}_txt{text2id[style[3]]}.lab' for i, style in enumerate(styles)]
            copy_ref_speech(src_wavs, src_txts, dst_wavs, dst_txts)
        else:
            print("1.1. Skip reference wav copy cause it has been Done!")
    else:
        print("1. Skip speech synthesis cause it has been Done!")

    ####### 2. Compute psd after mfa and save mfa/psd  #######
    enh_prosody_dict_json = os.path.join(out_res_dir, "prosody_{}_VS_{}.json".format(model1_n, model2_n))  # {"emo1": {"model1": {"psd/phone/speechid": [[], []]}}}
    if start_step <= 2 <= end_step:
        # Extract psd/MCD
        preprocess_config = configs1[0]
        prosody_dict = extract_psd(preprocess_config, model_name1, model_name2, out_dir)
        with open(enh_prosody_dict_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(prosody_dict, sort_keys=True, indent=4))
    else:
        with open(enh_prosody_dict_json, "r") as f:
            prosody_dict = json.load(f)
        print("2. Skip prosody extraction cause it has been done!")

    ####### 3. Vis attention map given attn file #######
    if start_step <= 3 <= end_step:
        with open(attn_dict_json, "r") as f:
            attn_dict = json.load(f)
        crossAttn_dict_for_vis = convert_attn_json(
            attn_dict, out_dir, show_t=0, show_b=0, show_h=0,
            show_txt=0)  # {"model1": {"emo1": np.array([syn_frames, ref_frames])}}}
        print("4. Vis attention map given attn file")
        vis_crossAttn(crossAttn_dict_for_vis, attn_dict, out_png, show_txt=0)

    ####### 4. Vis psd contour given json file  #######
    # save pitch_dict_for_vis
    if start_step <= 4 <= end_step:
        show_txt_id = 0
        pitch_dict_for_vis, energy_dict_for_vis = convert_vis_psd_json(prosody_dict,
                                                                       show_txt_id)  # for_vis: {"emo1": {"model1": list(psd_len)}}}
        vis_psd(pitch_dict_for_vis, energy_dict_for_vis, prosody_dict, out_png, show_txt_id)

    ####### 5. Stastic psd and mc given json file  #######
    psd_mcd_stat_res_json = os.path.join(out_res_dir,
                                         "psd_mcd_{}_VS_{}.json".format(model1_n,
                                                                        model2_n))  # {"emo1": {"model1": {"psd/phone/speechid": [[], []]}}}
    if start_step <= 5 <= end_step:
        psd_mcd_stat_res = statcz_psd_mcd(prosody_dict,
                                          out_dir)  # {"ang": {"modelA": [p1, e1, m1]}, "happy":{ "modelB": []]}}
        with open(psd_mcd_stat_res_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(psd_mcd_stat_res, sort_keys=True, indent=4))
        print(psd_mcd_stat_res)


def main_enh(styles: List[Tuple[Any, Any, Any, Any, Any, Any]],
        synTexts: List[str],
        model_name1: str,
        chk_pt1: str,
        configs1: Tuple[Any, Any, Any],
        out_dir: str = "option",  # middle result
        fs_model: int = 22050,
        start_step=1,
        end_step=2,
        enh_ind_syn_ref=(2, 2)):

    enh_ind_name = str(enh_ind_syn_ref[0]) + "_" + str(enh_ind_syn_ref[0])
    out_res_dir = out_dir + "/compareEnh_{}_VS_{}".format(None, enh_ind_name)

    out_png = out_res_dir + "/{}_similarity_{}.png"
    if not os.path.isdir(out_res_dir):
        Path(out_res_dir).mkdir(exist_ok=True, parents=True)

    enh_attn_dict_json = os.path.join(out_res_dir,
                                      "attn_{}_{}_VS_{}.json".format(model_name1, None,
                                                                         enh_ind_name))  # {"emo1": {"model1": {"speechid/qkdur/phone0": [[], []]}}}
    enh_prosody_dict_json = os.path.join(out_res_dir,
                                         "prosody_{}_{}_VS_{}.json".format(model_name1, None,
                                                                                        enh_ind_name))  # {"emo1": {"model1": {"psd/phone/speechid": [[], []]}}}
    # synthesis
    attn_enh_dict = syn_speech_from_model_enhance([model_name1, configs1, chk_pt1], styles, synTexts, out_dir,
                                                  fs_model=fs_model, enh_ind_syn_ref=enh_ind_syn_ref)
    with open(enh_attn_dict_json, "w", encoding="utf-8") as f:
        f.write(json.dumps(attn_enh_dict, sort_keys=True, indent=4))

    # extract psd
    preprocess_config = configs1[0]
    model_name1_noenh = model_name1 + "_None"
    model_name1_enh = model_name1 + "_" + enh_ind_name

    enh_prosody_dict = extract_psd(preprocess_config, model_name1_noenh, model_name1_enh, out_dir)
    with open(enh_prosody_dict_json, "w", encoding="utf-8") as f:
        f.write(json.dumps(enh_prosody_dict, sort_keys=True, indent=4))

    # vis_psd
    enh_show_txt_id = 0
    pitch_dict_for_vis, energy_dict_for_vis = convert_vis_psd_json(enh_prosody_dict,
                                                                   enh_show_txt_id)  # for_vis: {"emo1": {"model1": list(psd_len)}}}
    dur = 5  # temp
    aux_range = list(range(enh_ind_syn_ref[0], enh_ind_syn_ref[0] + dur))
    vis_psd_enh(pitch_dict_for_vis, enh_prosody_dict, out_png, enh_show_txt_id, aux_range=aux_range)

    # vis attention
    with open(enh_attn_dict_json, "r") as f:
        enh_attn_dict = json.load(f)
    enh_crossAttn_dict_for_vis = convert_attn_json(
        enh_attn_dict_json, out_dir, show_t=0, show_b=0, show_h=0,
        show_txt=0)  # {"model1": {"emo1": np.array([syn_frames, ref_frames])}}}
    print("4. Vis attention map given attn file")
    vis_crossAttn(enh_crossAttn_dict_for_vis, enh_attn_dict, out_png, show_txt=0)


def add_emo(eval_style_f, meta_json, eval_style_emo_f):
    """
    add emotion to eval_style file
    """
    with open(meta_json) as f:
        meta_dict = json.load(f)
    eval_style_emo_list = []
    with open(eval_style_f) as f:
        for l in f:
            speechid = l.split("|")[0]
            emo = meta_dict[speechid]["emotion"]
            eval_style_emo_list.append(l[:-1] + "|" + emo + "\n")
    with open(eval_style_emo_f, "w") as f:
        f.writelines(eval_style_emo_list)


if __name__ == "__main__":
    import argparse
    """
    model_config_input_extra
    model
        - stdit
        - stditCross: self + cross + FFT
        - stditMocha:
        - unet: unet model
        - dit: DiT model
        
        
        - stditCross_noguide_codec
        - stditCross_guideframe_codec
        - stditCross_guidephone_codec  # grad_200
        - stditCross_guideSyl_codec
        - stditCross_pguidephone_codec
        
        
    config (Only for stditCross)
        - base
        - qkNorm
        - monoHard: hard monotonic attention mask
        - guideLoss: guide crossAttention by monotonic matrix
        - phonemeRoPE: phoneme-level positional Embedding
        - refEncoder: mlp/vae/vaeGRL
    input
        - wav2vec2: or melstyle
        - codec: prosody
        - melVae: vae of melspectrogram
        - melVaeGRL: vae of melspectrogram with GRL to phoneme
    extra
        - noGlobal
    """
    config_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/config/ESD"

    # Config
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name1", type=str, default="stditCross_noguide_codec")
    #parser.add_argument("--model_name2", type=str, default="stditCross_guideLoss_codec")
    parser.add_argument("--model_name2", type=str, default="stditCross_guideframe_codec")
    parser.add_argument("--chk_pt1", type=str, default="grad_210.pt")
    parser.add_argument("--chk_pt2", type=str, default="grad_200.pt")
    parser.add_argument("--chk_pt3", type=str, default="chkpoint123.pt")
    parser.add_argument("--evalstyle", type=str, default="data/eval50_style1.txt") # evalstyle
    parser.add_argument("--evaltxt", type=str, default="data/eval50_paratxt1.txt") # evaltxt_para.txt
    parser.add_argument(
        "-p", "--preprocess_config", type=str,
        help="path to preprocess.yaml", default=config_dir + "/preprocess_stditCross.yaml")
    parser.add_argument(
        "-t", "--train_config", type=str,
        help="path to train.yaml", default=config_dir + "/train_stditCross.yaml")
    args = parser.parse_args()

    # add emotion
    meta_json = "data/metadata_new.json"
    eval_style_emo_f = args.evalstyle.split(".")[0] + "_emo.txt"
    #add_emo(args.evalstyle, meta_json, eval_style_emo_f)   # only for initiate

    # bone model config
    bone2configPath = {
        "gradtts": ("preprocess_gradTTS.yaml", "model_gradTTS_v3.yaml", "train_gradTTS.yaml"),
        "gradtts_cross": ("preprocess_gradTTS.yaml", "model_gradTTS_v3.yaml", "train_gradTTS.yaml"),
        "STDit": ("preprocess_styleAlignedTTS.yaml", "model_styleAlignedTTS.yaml", "train_styleAlignedTTS.yaml"),
        "stditCross": ("preprocess_stditCross.yaml", "model_stditCross.yaml", "train_stditCross.yaml")
    }
    vocoder_name = "hifigan"

    # VIS config
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
        "xy_label_sub": ("frame", "energy")
    }

    bone_n1, fineAdjNet_n1, input_n1 = args.model_name1.split("_")
    bone_n2, fineAdjNet_n2, input_n2 = args.model_name2.split("_")

    # fine-adjust config given bone_option, input_n
    model1_config1 = fine_adjust_configs(bone_n1, fineAdjNet_n1, bone2configPath, config_dir)
    model1_config2 = fine_adjust_configs(bone_n2, fineAdjNet_n2, bone2configPath, config_dir)

    # Cons
    train_file = ""

    # Convert input from args

    # Read INPUT
    ## emo, spk, wav_n, emo_tensor, melstyle_tensor, spk_tensor
    syn_styles = get_synStyle_from_file(eval_style_emo_f, split_char='|', melstyle_type="codec")  # emotion changed
    synTexts = get_synText_from_file(args.evaltxt)

    # OUTPUT
    out_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/exp/result5"

    """
    style_id = get_synText_from_file(args.evalstyle)
    from GradTTS.utils import parse_filelist

    filepaths_and_text = parse_filelist(train_file)
    # randomly choose style
    filepaths_and_text_choosed = filepaths_and_text[:20]
    # randomly choose text (para)
    text_choosed_para = [styles[-1] for styles in filepaths_and_text_choosed]
    text_choosed_unpara = []
    """
    # main
    if False:
        main(
            styles=syn_styles,
            synTexts=synTexts,
            model_name1=args.model_name1,
            model_name2=args.model_name2,
            chk_pt1=args.chk_pt1,
            chk_pt2=args.chk_pt2,
            configs1=model1_config1,
            configs2=model1_config2,
            out_dir=out_dir,
            vis_only=True,
            vis_config=pitch_vis_config,
            fs_model=22050,
            start_step=1,
            end_step=1,
        )

    # enh
    if True:
        enh_ind_syn_ref = (2, 2)
        main_enh(
            styles=syn_styles,
            synTexts=synTexts,
            model_name1=args.model_name1,
            chk_pt1=args.chk_pt1,
            configs1=model1_config1,
            out_dir=out_dir,
            fs_model=22050,
            start_step=1,
            end_step=1,
            enh_ind_syn_ref=enh_ind_syn_ref
        )


