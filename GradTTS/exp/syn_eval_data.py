
from scipy.io.wavfile import write
import os.path
import evaluate
import sys
sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS')
sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS/hifi-gan/')

import torch
from SER import predict_emotion
from GradTTS.inference_cond import inference_interp, inference
from visualization import show_psd, show_mel
from inference import inference_emodiff_interp, inference_fastspeech
import yaml
from GradTTS.model.utils import extract_pitch, pitchExtracdtion
from pathlib import Path
from visualization import show_psd, show_mel
from GradTTS.exp.util import get_pitch_match_score
import librosa
import numpy as np
from GradTTS.utils import get_emo_label

# constant parameter
#from GradTTS.const_param import emo_num_dict, emo_melstyle_dict, psd_dict, wav_dict
from GradTTS.const_param import logs_dir_par, logs_dir, config_dir, melstyle_dir, psd_dir, wav_dir


## FreqInterp
pitch_noPitch_dict = {
    "pitch_ref": [("Angry", "spk_id", ".npy"), ("Angry", "spk_id", ".npy")],
    "noPitch_ref": [("Angry", "spk_id", ".npy"), ("Angry", "spk_id", ".npy")]
}

"""
subdir/files in *_base_dir
"""

eval_gradtts_base_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/logs/"
eval_fastspeech_base_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/logs/"
eval_emodiff_base_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/logs/"

"""
eval_model_base_dir = {
    "interpTTS": eval_gradtts_base_dir + "gradtts_crossSelf_puncond_n1_neworder_fixmask",
    "emodiff": eval_emodiff_base_dir + "",
    "fastspeech": eval_fastspeech_base_dir + ""
}
"""
case_study_example_temp = {
    "ang_sad": "",
    "sad_hap": "",
    "ang_hap": ""}


"""
eval_interp_dir = eval_model_base_dir["interpTTS"]
interpNon_dir = eval_interp_dir + "/interpNon"
interpTemp_dir = eval_interp_dir + "/interpTemp"
interpFreq_dir = eval_interp_dir + "/interpFreq"

eval_emodiff_dir = eval_model_base_dir["emodiff"]
eval_fastsph_dir = eval_model_base_dir["fastspeech"]
"""


def synthesize_by_noInterp(emos=("Angry", "Sad", "Happy"),
                           syn_models=("interpTTS", "emodiff", "fastspeech"),
                           eval_dir="",
                           emo_melstyle_dict=dict(),
                           configs=None,
                           estimator_name=None,
                           chk_pt=None,
                           kwargs=None
                           ):
    for model_name in syn_models:
        if model_name == "interpTTS":
            # model-related configs
            for emo in emos:
                # style-related configs (from dictionary of specific emotions)
                emo_label = torch.LongTensor(get_emo_label(emo)).cuda()
                melstyle = (torch.from_numpy(np.load(melstyle_dir + "/" + emo_melstyle_dict[emo])).
                            unsqueeze(0).transpose(1, 2).cuda())
                # create subdir
                out_sub_dir = eval_dir + "/interpNon" + f"/{emo}"
                if not os.path.isdir(out_sub_dir):
                    Path(out_sub_dir).mkdir(exist_ok=True, parents=True)
                inference(
                    configs,
                    estimator_name,
                    chk_pt,
                    emo_label=emo_label,
                    melstyle=melstyle,
                    out_dir=out_sub_dir,
                    **kwargs,
                )
        elif model_name == "fastspeech":
            print("Please go to {}".format("/home/rosen/Project/FastSpeech2/exp_main.py"))
        elif model_name == "emodiff":
            pass


def synthesize_by_tempInterp(
        paired_emos=("Angry_Sad", "Sad_Happy", "Angry_Happy"),
        syn_models=("interpTTS", "emodiff", "fastspeech"),
        eval_interp_dir="",
        configs=None,
        estimator_name=None,
        chk_pt=None,
        kwargs=None
):
    """
    Synthesize tempInterp into subdir like:
        dir
            ang_sad
                1.wav
            sad_hap
    Args:
        paired_emos:
        syn_models:
    Returns:

    """
    for model_name in syn_models:
        if model_name == "interpTTS":
            # model-related configs
            for pair_emo in paired_emos:
                # style-related configs (from dictionary of specific emotions)
                emo1, emo2 = pair_emo.split("_")
                emo_label1 = torch.LongTensor(get_emo_label(emo1)).cuda()
                emo_label2 = torch.LongTensor(get_emo_label(emo2)).cuda()
                melstyle1 = (torch.from_numpy(np.load(melstyle_dir + "/" + emo_melstyle_dict[emo1])).
                             unsqueeze(0).transpose(1, 2).cuda())
                melstyle2 = (torch.from_numpy(np.load(melstyle_dir + "/" + emo_melstyle_dict[emo2])).
                             unsqueeze(0).transpose(1, 2).cuda())

                # create subdir
                out_sub_dir = eval_interp_dir + "/interpTemp" + f"/{pair_emo}"
                if not os.path.isdir(out_sub_dir):
                    Path(out_sub_dir).mkdir(exist_ok=True, parents=True)

                inference_interp(
                    configs,
                    estimator_name,
                    chk_pt,
                    emo_label1=emo_label1,
                    melstyle1=melstyle1,
                    emo_label2=emo_label2,
                    melstyle2=melstyle2,
                    out_dir=out_sub_dir,
                    **kwargs,
                )

        elif model_name == "emodiff":
            # model-related configs
            model_config, kwargs = get_configs(model_name)
            for emo in emos:
                # style-related configs
                emo1, emo2 = emo.split("_")
                inference_emodiff_interp(
                    model_config,
                    **kwargs,
                    emo1=emo1,
                    emo2=emo2
                )

        elif model_name == "fastspeech":
            print("Please go to {}".format("/home/rosen/Project/FastSpeech2/exp_main.py"))

        else:
            print("please choose model from emodiff, fastspeech, or interptts")


def synthesize_by_freInterp(freq_area="pitch",
                            syn_models=("interpTTS", "emodiff", "fastspeech")
                            ):
    """??? Next step ???"""
    for model_name in syn_models:
        if model_name == "interpTTS":
            # Model-related configs
            config_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/config/ESD"
            preprocess_config = yaml.load(
                open(config_dir + "/preprocess_gradTTS.yaml", "r"), Loader=yaml.FullLoader
            )
            model_config = yaml.load(open(
                config_dir + "/model_gradTTS.yaml", "r"), Loader=yaml.FullLoader)
            train_config = yaml.load(open(
                config_dir + "/train_gradTTS.yaml", "r"), Loader=yaml.FullLoader)
            configs = (preprocess_config, model_config, train_config)

            ## Model version
            checkpoint = "grad_54.pt"
            chk_pt = exp_dir + checkpoint

            ## Model hyper
            time_steps = 100
            mask_time_step = int(time_steps / 1.0)
            mask_all_layer = True
            guidence_strength = 3.0   # classifer-guidence strength
            chpt_n = checkpoint.split(".")[0].split("_")[1]

            # output
            out_dir = f"{exp_dir}freq_eval/interpTTS"
            if not os.path.isdir(out_dir):
                Path(out_dir).mkdir(exist_ok=True)

            # style-related configs (from dictionary of specific emotions)
            if freq_area == "pitch":
                ref1_list = pitch_noPitch_dict["pitch_ref"]
                ref2_list = pitch_noPitch_dict["noPitch_ref"]
                syn_txt = "resources/filelists/synthesis1.txt"
                for ref1, ref2 in zip(ref1_list, ref2_list):
                    emo1, spk1, mel1, audio1 = ref1
                    emo2, spk2, mel2, audio2 = ref2
                    assert spk1 == spk2   # ?? should be same ??
                    inference_interp(
                        configs,
                        "gradtts_cross",
                        chk_pt,
                        syn_txt=syn_txt,
                        time_steps=time_steps,
                        spk=spk1,
                        emo_label1=emo1,
                        melstyle1=mel1,
                        pitch1=audio1,
                        emo_label2=emo2,
                        melstyle2=mel2,
                        pitch2=audio2,
                        out_dir=out_dir,
                        interp_type="freq",
                        mask_time_step=mask_time_step,
                        mask_all_layer=mask_all_layer,
                        guidence_strength=guidence_strength
                    )
            elif model_name == "emodiff":
                out_dir = f"{exp_dir}freq_eval/emodiff" # outdir/sample{i}.wav
                pass
            elif model_name == "fastspeech":
                out_dir = f"{exp_dir}freq_eval/fastspeech" # outdir/sample{i}.wav
                pass
            else:
                print("please choose model from emodiff, fastspeech, or interptts")



def get_configs(model_name, logs_dir):
    """
    Get model-related configs
    """
    if model_name == "interpTTS":
        config_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/config/ESD"
        preprocess_config = yaml.load(
            open(config_dir + "/preprocess_gradTTS.yaml", "r"), Loader=yaml.FullLoader
        )
        model_config = yaml.load(open(
            config_dir + "/model_gradTTS_v2.yaml", "r"), Loader=yaml.FullLoader)
        train_config = yaml.load(open(
            config_dir + "/train_gradTTS.yaml", "r"), Loader=yaml.FullLoader)
        configs = (preprocess_config, model_config, train_config)
        estimator_name = "gradtts_cross"

        # ckpt version
        checkpoint = "/models/grad_65.pt"   # grad_54.pt
        chk_pt = logs_dir + checkpoint

        # text
        syn_txt = "../resources/filelists/synthesis3.txt"

        # inference related
        time_steps = 100
        guidence_strength = 3.0

        # style related
        speaker_id = 19
        spk_tensor = torch.LongTensor([speaker_id]).cuda() if not isinstance(speaker_id, type(None)) else None
        ## interpolated related
        interp_type = "temp"
        mask_time_step = int(time_steps / 1.0)
        mask_all_layer = True
        temp_mask_value = 0

        kwargs = {
            "syn_txt": syn_txt,
            "time_steps": time_steps,
            "spk": spk_tensor,
            "interp_type": interp_type,
            "mask_time_step": mask_time_step,
            "mask_all_layer": mask_all_layer,
            "temp_mask_value": temp_mask_value,
            "guidence_strength": guidence_strength
        }

        return configs, estimator_name, chk_pt, kwargs
    elif model_name == "":
        return ""
    elif model_name == "":
        return ""