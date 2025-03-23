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

from GradTTS.exp.utils import convert_vis_psd_json, convert_attn_json, copy_ref_speech, fine_adjust_configs, get_synStyle_from_file
from GradTTS.exp.extract_psd import extract_psd
from GradTTS.exp.syn_speech import syn_speech_from_models
from GradTTS.exp.visualization import vis_psd, vis_crossAttn
from GradTTS.exp.utils import get_synText_from_file



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
        out_dir: str = "option",
        vis_only: bool = True,
        vis_config: dict = {"row_col_num": (2, 3)},
        fs_model: int = 22050,
        out_png = "result/{}_similarity_{}.png",
        start_step=1,
        end_step=2
):
    ####### 1. Synthesize speech and save attn file #######
    if start_step <= 1 <= end_step:
        syn_speech_from_models([model_name1, configs1, chk_pt1],
                               [model_name2, configs2, chk_pt2],
                               styles, synTexts, out_dir)
        out_ref_dir = os.path.join(out_dir, "reference")
        if not os.path.isdir(out_ref_dir):
            print("1.1. Do reference wav copy!")
            Path(out_ref_dir).mkdir(exist_ok=True)
            src_wavs = [style[2] for style in styles]
            src_txts = [style[3] for style in styles]
            dst_wavs = [f'{out_ref_dir}/spk{style[1]}_{style[0]}_txt{i}.wav' for i, style in enumerate(styles)]
            dst_txts = [f'{out_ref_dir}/spk{style[1]}_{style[0]}_txt{i}.lab' for i, style in enumerate(styles)]
            copy_ref_speech(src_wavs, src_txts, dst_wavs, dst_txts)
        else:
            print("1.1. Skip reference wav copy cause it has been Done!")
    else:
        print("1. Skip speech synthesis cause it has been Done!")

    ####### 2. Compute psd after mfa and save mfa/psd  #######
    prosody_dict_json = os.path.join(out_dir, "prosody_dict.json")  # MetaData {"emo1": {"model1": {"psd1/phone": []}}}
    if start_step <= 2 <= end_step:
        # Extract psd/MCD
        prosody_dict = extract_psd(configs1[0], model_name1, model_name2, out_dir)
        with open(prosody_dict_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(prosody_dict, sort_keys=True, indent=4))
    else:
        with open(prosody_dict_json, "r") as f:
            prosody_dict = json.load(f)
        print("2. Skip prosody extraction cause it has been done!")

    ####### 3. Vis psd contour given json file  #######
    pitch_dict_for_vis = convert_vis_psd_json(prosody_dict)  # for_vis: {"emo1": {"model1": list(psd_len)}}}
    # save pitch_dict_for_vis

    if start_step <= 3 <= end_step:
        vis_psd(pitch_dict_for_vis, prosody_dict, out_png)

    ####### 4. Vis attention map given attn file #######
    crossAttn_dict_for_vis = convert_attn_json(
        prosody_dict, out_dir, show_t=0, show_b=0, show_h=0, show_txt=0)  # {"model1": {"emo1": np.array([syn_frames, ref_frames])}}}
    if start_step <= 4 <= end_step:
        print("4. Vis attention map given attn file")
        vis_crossAttn(crossAttn_dict_for_vis, prosody_dict, out_png, show_txt=0)


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
    parser.add_argument("--model_name1", type=str, default="stditCross_base_codec")
    parser.add_argument("--model_name2", type=str, default="stditCross_guideLoss_codec")
    parser.add_argument("--model_name3", type=str, default="stditCross_phonemeRoPE_codec")
    parser.add_argument("--chk_pt1", type=str, default="grad_110.pt")
    parser.add_argument("--chk_pt2", type=str, default="grad_180.pt")
    parser.add_argument("--chk_pt3", type=str, default="chkpoint123.pt")
    parser.add_argument("--evalstyle", type=str, default="evalstyle.txt")
    parser.add_argument("--evaltxt", type=str, default="evaltxt_para.txt")
    parser.add_argument(
        "-p", "--preprocess_config", type=str,
        help="path to preprocess.yaml", default=config_dir + "/preprocess_stditCross.yaml")
    parser.add_argument(
        "-t", "--train_config", type=str,
        help="path to train.yaml", default=config_dir + "/train_stditCross.yaml")
    args = parser.parse_args()

    # bone model config
    models_config_dict = {
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

    bone_n1, config_n1, input_n1 = args.model_name1.split("_")
    bone_n2, config_n2, input_n2 = args.model_name2.split("_")

    # fine-adjust config given bone_option, input_n
    model1_config1 = fine_adjust_configs(bone_n1, config_n1, models_config_dict, config_dir)
    model1_config2 = fine_adjust_configs(bone_n2, config_n2, models_config_dict, config_dir)

    # Conts
    train_file = ""

    # Convert input from args

    # Read INPUT
    ## emo, spk, wav_n, emo_tensor, melstyle_tensor, spk_tensor
    syn_styles = get_synStyle_from_file(args.evalstyle, split_char='|', melstyle_type="codec")  # emotion changed
    synTexts = get_synText_from_file(args.evaltxt)

    # OUTPUT
    out_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/exp/result"
    out_res_dir = out_dir + "/result_stditCompare"
    out_png = out_res_dir + "/{}_similarity_{}.png"

    if not os.path.isdir(out_res_dir):
        Path(out_res_dir).mkdir(exist_ok=True, parents=True)

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
        out_png=out_png,
        start_step=4,
        end_step=4,
    )