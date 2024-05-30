"""
# Temp_interpolation
- 1st/2nd harf Emotion detection
- 1st/2nd harf Preference test
- Example study

# Freq_interpolation
- ref1/ref2/interp PSD matching
- Example study

# Non_interpolation
- EmoPrefer/MOS with p_uncond/weight
"""
from scipy.io.wavfile import write
import os.path
import evaluate
import sys
sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS')
sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS/hifi-gan/')

import torch
from SER import predict_emotion
from GradTTS.inference_cond import inference_interp
from visualization import show_psd, show_mel
from inference import inference_emodiff_interp, inference_fastspeech
import yaml
from GradTTS.model.utils import extract_pitch, pitchExtracdtion
from pathlib import Path
from visualization import show_psd, show_mel
from GradTTS.exp.util import get_pitch_match_score
import librosa
import numpy as np
from syn_eval_data import synthesize_by_tempInterp, synthesize_by_noInterp

from GradTTS.const_param import emo_num_dict, emo_melstyle_dict1, emo_melstyle_dict, psd_dict, wav_dict
from GradTTS.const_param import logs_dir_par, logs_dir, config_dir, melstyle_dir, psd_dir, wav_dir
from GradTTS.const_param import label2id_SER


# Referance speech
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

eval_model_base_dir = {
    #"interpTTS": eval_gradtts_base_dir + "interpEmoTTS_frame2frameAttn_noJoint",
    "interpTTS": eval_gradtts_base_dir + "interpEmoTTS_linearAttn",
    #"interpTTS": eval_gradtts_base_dir + "gradtts_crossSelf_puncond_n1_neworder_fixmask",
    "emodiff": eval_emodiff_base_dir + "",
    "fastspeech": eval_fastspeech_base_dir + ""}

case_study_example_temp = {
    "ang_sad": "",
    "sad_hap": "",
    "ang_hap": ""}


def combineStyleToken(token1, token2):
    token12 = token1 + token2
    return token12


def combineMel(mel1, mel2):
    mel12 = mel1 + mel2
    return mel12


def combinePSD(psd1, psd2):
    psd12 = psd1 + psd2
    return psd12


model_yaml = {
    "crossAttn_f2f_nonJoint": "model_gradTTS_v2",
    "crossAttn_f2f": "model_gradTTS",
    "crossAttn_f2b": "",  # Change to Multihead2
    "linearAttn": "model_gradTTS_linear"
}

def get_interpTTS_infer_config(logs_dir, infer_type="interp", sub_model_type="crossAttn_f2f_nonJoint"):
    """
    Get model-related configs
    """
    config_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/config/ESD"
    preprocess_config = yaml.load(
        open(config_dir + "/preprocess_gradTTS.yaml", "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(
        config_dir + "/{}.yaml".format(model_yaml[sub_model_type]), "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        config_dir + "/train_gradTTS.yaml", "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    estimator_name = "gradtts_cross"

    # ckpt version
    checkpoint = "/models/grad_118.pt"   # grad_54.pt
    chk_pt = logs_dir + checkpoint

    # text
    syn_txt = "../resources/filelists/synthesis1.txt"

    # inference related
    time_steps = 50
    guidence_strength = 3.0

    # spk related
    speaker_id = 19
    spk_tensor = torch.LongTensor([speaker_id]).cuda() if not isinstance(speaker_id, type(None)) else None

    # interpolated related
    interp_type = "temp"
    mask_time_step = int(time_steps / 1.0)
    mask_all_layer = True
    temp_mask_value = 0

    if infer_type == "interp":
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
    else:
        kwargs = {
            "syn_txt": syn_txt,
            "time_steps": time_steps,
            "spk": spk_tensor,
            "guidence_strength": guidence_strength
        }
    return configs, estimator_name, chk_pt, kwargs

def get_emoDiff_infer_config():
    pass


class Non_interp_eval():
    def __init__(self,
                 emos=("Angry", "Happy", "Sad"),
                 syn_models=("interpTTS", "emodiff", "fastspeech"),
                 need_syn=False,
                 sub_model_type="linearAttn"
                 ):
        configs, estimator_name, chk_pt, kwargs = get_interpTTS_infer_config(
            eval_model_base_dir["interpTTS"],
            infer_type="nonInterp",
            sub_model_type=sub_model_type
        )
        if "interpTTS" in syn_models:
            if need_syn:
                synthesize_by_noInterp(
                    emos=emos,
                    syn_models=syn_models,
                    eval_dir=eval_model_base_dir["interpTTS"],
                    emo_melstyle_dict=emo_melstyle_dict1,
                    configs=configs,
                    estimator_name=estimator_name,
                    chk_pt=chk_pt,
                    kwargs=kwargs
                )

    def eval_by_mcd(self,
                    syn_models = ("modelA")
                    ):
        mcd_dict = dict() # {"modelA": [()]}
        if "interpTTS" in syn_models:
            mcd_interpTTS_dict = evaluate_mcd_result(self.interpTemp_dir)  # {"ang": [1.2, 2.3], "hap": [2.3, ...]}
        elif "gradTTS" in syn_models:
            gradTTS_dir = ""
            mcd_gradTTS_dict = evaluate_mcd_result(gradTTS_dir)
        return mcd_dict

    def eval_mos(self,
                 punda=(0.2, 0.8),
                 weight=(0.3, 0.7)
                 ):
        """read mos result to dict"""
        mos_dict = dict()
        for p in punda:
            for w in weight:
                pass

    # generate evaluation data for mos
    # generate evaluation data for smos
    def eval_smos(self):
        """read smos result to dict"""
        pass

    def eval_emoPref(self, punda=(0.2, 0.8), weight=(0.3, 0.7)):
        """???"""
        for p in punda:
            for w in weight:
                pass

    # Visualization


class InterpEval:
    def __init__(self):
        """
        Synthesize speech from proposed/benchmarked models and evaluate them.
        generate temp interpolation by given below two emotions:
        Ang -> Sad
        Sad -> Hap
        Ang -> Hap
        """
        # Used for evaluating tempInterp
        # Used for evaluating freqInterp
        # evaluation dir
        self.eval_interp_dir = eval_model_base_dir["interpTTS"]
        self.interpTemp_dir = self.eval_interp_dir + "/interpTemp"
        self.interpFreq_dir = self.eval_interp_dir + "/interpFreq"
        self.eval_emodiff_dir = eval_model_base_dir["emodiff"]
        self.eval_fastsph_dir = eval_model_base_dir["fastspeech"]

    def eval_by_SER(self,
                    paired_emos=("Angry_Sad", "Sad_Hap", "Angry_Happy"),
                    syn_models=("interpTTS", "emodiff", "fastspeech"),
                    need_syn=False
                    ):
        """
        detect 1st/2nd half emotion by SER
        Returns:
        """
        configs, estimator_name, chk_pt, kwargs = get_interpTTS_infer_config(
            eval_model_base_dir["interpTTS"])

        result = dict()  # {"modelA": {"emo1_emo2": [emo1_acc, emo2_acc],}}
        if "interpTTS" in syn_models:
            if need_syn:
                synthesize_by_tempInterp(
                    paired_emos=paired_emos,
                    syn_models=syn_models,
                    eval_interp_dir=eval_model_base_dir["interpTTS"],
                    configs=configs,
                    estimator_name=estimator_name,
                    chk_pt=chk_pt,
                    kwargs=kwargs
                )
            audio_pred_dict, acc_res = evalute_ser_result(self.interpTemp_dir)
            result["interpTTS"] = [audio_pred_dict, acc_res]
        # write in FastSpeech2/exp_main.py
        elif "fastspeech" in syn_models:
            syn_dir = "/home/rosen/Project/FastSpeech2/evaluation/interpTemp_fspeechGst"
            # check if dir is corrected formated
            audio_pred_dict, acc_res = evalute_ser_result(syn_dir)
            result["fastspeech"] = [audio_pred_dict, acc_res]
        elif "emodiff" in syn_models:
            syn_dir = "/home/rosen/Project/FastSpeech2/evaluation/???"
            audio_pred_dict, acc_res = evalute_ser_result(syn_dir)
            result["emodiff"] = [audio_pred_dict, acc_res]
        return result

    def eval_by_EmoAccContour(self,
                              model_evalDir_dict=("interpTTS", "emodiff", "fastspeech"),
                              sep_nums=4
                              ):
        """"""
        result = dict()  # {"emo1_emo2": {"modelA": [(emo1_t1_acc, emo1_t2_acc), (emo2_..)],}}
        return result

    def eval_by_pref(self,
                     pref_num=5):
        """???"""
        audio_gd_dict = get_audio_gd(self.exp_outdir)
        audio_gd_pref_dict = filter_pref_audio(audio_gd_dict, pref_num=pref_num)
        ## create preference test folder
        audio_gd_pref_dir = create_pref_dir(audio_gd_pref_dict)

        ## DO the Experiment

        ## Collect result

    def eval_by_pitchDetect(self,
                            sampling_rate=16000,
                            hop_length=256,
                            need_trim=False
                            ):
        """
        Detect pitch matching
        Returns:
        """
        # get ref pitch
        _, _, _, ref_audio = pitch_noPitch_dict["pitch_ref"]
        ref_pitch = extract_pitch(
                ref_audio,
                sampling_rate=sampling_rate,
                hop_length=hop_length
            )
        # get syn pitch
        pitch_list = []
        for audio in os.listdir(self.exp_dir):
            audio_path = os.path.join(self.exp_dir, audio)
            pitch = extract_pitch(
                    audio_path,
                    sampling_rate=sampling_rate,
                    hop_length=hop_length
                )
            pitch_list.append(pitch)

        # pitch matching score ?
        pitch_score = get_pitch_match_score(ref_pitch, pitch_list[0])

        # pitch matching figure ?
        choose_pitch_id = 3
        out_png = self.exp_dir + "freq_eval/interpTTS/pitch_match.png"
        show_psd(pitch_list[3], ref_pitch)

    def eval_exampleCase(self):
        """

        Returns:

        """
        audios = [case_study_example_temp["ang_hap"],
                  case_study_example_temp["ang_sad"],
                  case_study_example_temp["sad_hap"]]
        show_mel(audios)


def evalute_ser_result(syn_dir):
    audio_gd_dict = get_audio_gd(syn_dir)  # {"ang_hap": ["1.wav", "2.wav", ...], ...}
    audio_pred_dict = get_twohalf_emo(audio_gd_dict,
                                      syn_dir)  # {"ang_hap": [("ang", "hap"), ("hap", "hap"), ...], ...}
    acc_res = get_acc_res(audio_pred_dict)  # {"ang_sad": (0.8, 0.9), ...}
    return audio_pred_dict, acc_res

def get_emo_label(emo: str, emo_value=1):
    emo_emb_dim = len(emo_num_dict.keys())
    emo_label = [[0] * emo_emb_dim]
    emo_label[0][emo_num_dict[emo]] = emo_value
    return emo_label

def evaluate_mcd_result(syn_dir):
    ref_dir = ""
    audio_gd_dict = get_audio_gd(syn_dir)  # {"ang": ["1.wav", "2.wav", ...], "hap": ["1.wav", "2.wav", ...]}
    ref_gd_dict = get_audio_ref(ref_dir)     # {"ang": ["1.wav", "2.wav", "3.wav", ...], "hap": [".."] }
    mcd_dict = evaluate_mcd_result()  # {}   # {"ang": [1.2, 2.3], "hap": [2.3, ...]}
    return mcd_dict


def get_twohalf_emo(audio_gd_dict, parent_dir):
    """

    Args:
        audio_gd_dict:
            {"ang_hap": ["1.wav", "2.wav", ...], ...}
    Returns:
        audio_predEmo_dict:
            {"ang_hap": [("ang", "hap"), ("hap", "hap"),
                         ...], ...}
    """
    audio_predEmo_dict = {}
    for sub_dir, audio_f_list in audio_gd_dict.items():
        audio_predEmo_dict[sub_dir] = []
        for audio_f in audio_f_list:
            if not audio_f.endswith(".wav"):
                continue
            audio_f_path = parent_dir + "/" + sub_dir + "/" + audio_f
            y, _ = librosa.load(audio_f_path, sr=16000)
            first_y = y[:int(len(y)/2.0)]
            second_y = y[int(len(y)/2.0):]

            # half_dir
            half_dir = parent_dir + "/" + sub_dir + "/half/"
            if not os.path.isdir(half_dir):
                Path(half_dir).mkdir()

            audio_f_1st = half_dir + audio_f[:-4] + "_1st.wav"
            audio_f_2nd = half_dir + audio_f[:-4] + "_2nd.wav"
            write(audio_f_1st, 16000, first_y)
            write(audio_f_2nd, 16000, second_y)

            emo_dict1, pred_emo1 = predict_emotion(audio_f_1st, 16000)
            emo_dict2, pred_emo2 = predict_emotion(audio_f_2nd, 16000)
            audio_predEmo_dict[sub_dir].append((pred_emo1, pred_emo2))
    return audio_predEmo_dict


def get_audio_gd(interpTemp_dir):
    """
    Args:
        interpTemp_dir:
    Returns:
        audio_gd_dict:

    """
    audio_gd_dict = {}
    for sub_dir in os.listdir(interpTemp_dir):
        audio_gd_dict[sub_dir] = []
        for wav_f in os.listdir(interpTemp_dir + "/" + sub_dir):
            if wav_f.endswith(".wav"):
                audio_gd_dict[sub_dir].append(wav_f)
    return audio_gd_dict


def get_acc_res(audio_predEmo_dict):
    """
    audio_predEmo_dict:
        {"ang_hap": [("ang", "hap"), ("hap", "hap"), ...], ...}
    Returns:
        acc_res:
            {"ang_sad": (0.8, 0.9), ...}
    """

    acc_metric = evaluate.load("accuracy")
    acc_res = {}

    for sub_dir, pred_twoEmo in audio_predEmo_dict.items():
        pred_1st_emos = [label2id_SER[emos[0]] for emos in pred_twoEmo]
        pred_2nd_emos = [label2id_SER[emos[1]] for emos in pred_twoEmo]
        gd_1st_emos = [emo_num_dict[sub_dir.split("_")[0]]] * len(pred_1st_emos)
        gd_2nd_emos = [emo_num_dict[sub_dir.split("_")[1]]] * len(pred_1st_emos)

        acc_1st_emos = acc_metric.compute(predictions=pred_1st_emos, references=gd_1st_emos)["accuracy"]
        acc_2nd_emos = acc_metric.compute(predictions=pred_2nd_emos, references=gd_2nd_emos)["accuracy"]
        print(acc_1st_emos, acc_2nd_emos)
        acc_res[sub_dir] = (acc_1st_emos, acc_2nd_emos)
    return acc_res


def filter_pref_audio():
    pass


def create_pref_dir(audio_gd_pref_dict):
    pass



###################### NOT USED ################
"""
def combineStyle(combineType="style_token", style1="", style2="", fastspeech=None):
    if combineType == "style_token" or combineType == "psd":
        if fastspeech == None:
            raise IOError("Please input fastpseech model")

    if combineType == "style_token":
        gst_token1, _ = fastspeech(style1)
        gst_token2, _ = fastspeech(style2)
        gst_token12 = combineStyleToken(gst_token1, gst_token2)
        return gst_token12
    elif combineType == "mel":
        combineMel(style1, style2)
    elif combineType == "psd":
        _, psd1 = fastspeech(style1)
        gst_token2, _ = fastspeech(style2)
        gst_token12 = combineStyleToken(gst_token1, gst_token2)
        return gst_token12()
"""


if __name__ == '__main__':

    # Non_Interp
    EVAL_NON_INTERP = True
    NEED_SYN = True

    # temp_Interp
    EVAL_TEMP_INTERP = False
    NEED_SYN_TEMPINTERP = False
    SER_EVAL, PREF_EVAL, TEMP_CASE_EVAL = True, False, False

    # freq_Interp
    EVAL_FREQ_INTERP = False
    NEED_SYN_FREQINTERP = False
    PSD_MATCH_EVAL, FREQ_CASE_EVAL = False, False

    if EVAL_NON_INTERP:
        # input
        p_unda, weight = (0.2, 0.8), (0.3, 0.7)
        no_eval = Non_interp_eval(emos=("Angry", "Happy", "Sad"),
                                  syn_models=("interpTTS", ),
                                  need_syn=NEED_SYN
                                  )
        # MCD
        no_eval.eval_by_mcd()
        no_eval.eval_mos()
        no_eval.eval_smos()
        # Case study
        no_eval.eval_emoPref()

    temp_eval = InterpEval()
    if EVAL_TEMP_INTERP:
        # sub evaluation
        if SER_EVAL:
            res_acc = temp_eval.eval_by_SER(
                paired_emos=("Angry_Sad", "Sad_Happy", "Angry_Happy"),
                syn_models=("interpTTS",),
                #syn_models=("fastspeech",),
                need_syn=NEED_SYN_TEMPINTERP)
        if PREF_EVAL:
            temp_eval.eval_by_pref()
        # Case study
        if TEMP_CASE_EVAL:
            temp_eval.eval_exampleCase()

    if EVAL_FREQ_INTERP:
        if NEED_SYN_FREQINTERP:
            temp_eval.synthesize_by_freInterp()
        # PSD match test
        if PSD_MATCH_EVAL:
            temp_eval.eval_by_pitchDetect()
        # Case study
        if FREQ_CASE_EVAL:
            temp_eval.eval_exampleCase()