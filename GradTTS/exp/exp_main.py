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

# Referance speech
## FreqInterp
emo_num_dict = {
    "Angry": 0,
    "Surprise": 1,
    "Sad": 2,
    "Neutral": 3,
    "Happy": 4
}
## SER dataset
label2id_SER = {
    "angry": 0,
    "calm": 3,
    "disgust": 0,
    "fearful": 5,
    "happy": 4,
    "neutral": 3,
    "sad": 2,
    "surprised": 1
}

emo_melstyle_dict = {
    "Angry": "0019_000401.npy",
    "Surprise": "0019_001451.npy",
    "Sad": "0019_001101.npy",
    "Neutral": "0019_000051.npy",
    "Happy": "0019_000751.npy"
}

## FreqInterp
pitch_noPitch_dict = {
    "pitch_ref": [("Angry", "spk_id", ".npy"), ("Angry", "spk_id", ".npy")],
    "noPitch_ref": [("Angry", "spk_id", ".npy"), ("Angry", "spk_id", ".npy")]
}
f4_noF4_dict = {

}

"""
subdir/files in *_base_dir

"""
eval_gradtts_base_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/logs/"
eval_fastspeech_base_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/logs/"
eval_emodiff_base_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/logs/"

eval_model_base_dir = {
    "interpTTS": eval_gradtts_base_dir + "gradtts_crossSelf_puncond_n1_neworder_fixmask",
    "emodiff": eval_emodiff_base_dir + "",
    "fastspeech": eval_fastspeech_base_dir + ""
}

case_study_example_temp = {
    "ang_sad": "",
    "sad_hap": "",
    "ang_hap": ""
}


def combineStyleToken(token1, token2):
    token12 = token1 + token2
    return token12


def combineMel(mel1, mel2):
    mel12 = mel1 + mel2
    return mel12


def combinePSD(psd1, psd2):
    psd12 = psd1 + psd2
    return psd12


def combineStyle(combineType="style_token", style1="", style2="", fastspeech=None):
    """
    
    Args:
        combineType (): 
            style_token:
            mel:
            psd:

    Returns:

    """
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
            config_dir + "/model_gradTTS.yaml", "r"), Loader=yaml.FullLoader)
        train_config = yaml.load(open(
            config_dir + "/train_gradTTS.yaml", "r"), Loader=yaml.FullLoader)
        configs = (preprocess_config, model_config, train_config)
        model_name = "gradtts_cross"

        checkpoint = "/grad_54.pt"
        chk_pt = logs_dir + checkpoint
        syn_txt = "../resources/filelists/synthesis3.txt"
        time_steps = 100
        speaker_id = 19
        spk_tensor = torch.LongTensor([speaker_id]).cuda() if not isinstance(speaker_id, type(None)) else None


        interp_type = "temp"
        mask_time_step = int(time_steps / 1.0)
        mask_all_layer = True
        temp_mask_value = 0
        guidence_strength = 3.0

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

        return configs, model_name, chk_pt, kwargs
    elif model_name == "":
        return ""
    elif model_name == "":
        return ""


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

    def synthesize_by_freInterp(self,
                                freq_area="pitch",
                                syn_models=("interpTTS", "emodiff", "fastspeech")
                                ):
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
                chk_pt = self.exp_dir + checkpoint

                ## Model hyper
                time_steps = 100
                mask_time_step = int(time_steps / 1.0)
                mask_all_layer = True
                guidence_strength = 3.0   # classifer-guidence strength
                chpt_n = checkpoint.split(".")[0].split("_")[1]

                # output
                out_dir = f"{self.exp_dir}freq_eval/interpTTS"
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
                    out_dir = f"{self.exp_dir}freq_eval/emodiff" # outdir/sample{i}.wav
                    pass
                elif model_name == "fastspeech":
                    out_dir = f"{self.exp_dir}freq_eval/fastspeech" # outdir/sample{i}.wav
                    pass
                else:
                    print("please choose model from emodiff, fastspeech, or interptts")
    def synthesize_by_tempInterp(self,
                                 paired_emos=("Angry_Sad", "Sad_Happy", "Angry_Happy"),
                                 syn_models=("interpTTS", "emodiff", "fastspeech")
                                 ):
        """
        Synthesize tempInterp into subdir like:
            dir
                ang_sad
                    1.wav
                    2.wav
                    ...
                sad_hap
        Args:
            paired_emos:
            syn_models:

        Returns:

        """
        melstyle_dir = "/home/rosen/Project/FastSpeech2/preprocessed_data/ESD/emo_reps"

        for model_name in syn_models:
            if model_name == "interpTTS":
                # model-related configs
                configs, estimator_name, chk_pt, kwargs = get_configs(model_name, logs_dir=self.eval_interp_dir)

                for pr_emos in paired_emos:
                    # style-related configs (from dictionary of specific emotions)
                    emo1, emo2 = pr_emos.split("_")
                    emo_label1 = torch.LongTensor(get_emo_label(emo1)).cuda()
                    emo_label2 = torch.LongTensor(get_emo_label(emo2)).cuda()
                    melstyle1 = (torch.from_numpy(np.load(melstyle_dir + "/" + emo_melstyle_dict[emo1])).
                                 unsqueeze(0).transpose(1, 2).cuda())
                    melstyle2 = (torch.from_numpy(np.load(melstyle_dir + "/" + emo_melstyle_dict[emo2])).
                                 unsqueeze(0).transpose(1, 2).cuda())

                    # create subdir
                    out_sub_dir = self.interpTemp_dir + f"/{pr_emos}"
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
                for pr_emos in self.paired_emos:
                    # style-related configs
                    emo1, emo2 = pr_emos.split("_")
                    inference_emodiff_interp(
                        model_config,
                        **kwargs,
                        emo1=emo1,
                        emo2=emo2
                    )

            elif model_name == "fastspeech":
                # model-related configs
                for pr_emos in self.paired_emos:
                    # style-related configs
                    emo1, emo2 = pr_emos.split("_")
                    style12 = combineStyle(
                        combineType="mel",
                        style1=emo_melstyle_dict[emo1],
                        style2=emo_melstyle_dict[emo1]
                    )
                    inference_fastspeech(
                        style=style12
                    )
            else:
                print("please choose model from emodiff, fastspeech, or interptts")

    def eval_by_SER(self,
                    paired_emos=("Angry_Sad", "Sad_Hap", "Angry_Happy"),
                    syn_models=("interpTTS", "emodiff", "fastspeech"),
                    need_syn=False,
                    syn_dir=None
                    ):
        """
        detect 1st/2nd half emotion by SER
        Returns:

        """
        result = {}
        if "interpTTS" in syn_models:
            if need_syn:
                self.synthesize_by_tempInterp(
                    paired_emos=paired_emos,
                    syn_models=syn_models
                )
            audio_pred_dict, acc_res = evalute_ser_result(self.interpTemp_dir)
            result["interpTTS"] = [audio_pred_dict, acc_res]
        elif "fastspeech" in syn_models:
            syn_dir = "/home/rosen/Project/FastSpeech2/evaluation/interpTemp_fspeechGst"
            # check if dir is corrected formated
            audio_pred_dict, acc_res = evalute_ser_result(syn_dir)
            result["fastspeech"] = [audio_pred_dict, acc_res]
        elif "emodiff" in syn_models:
            syn_dir = "/home/rosen/Project/FastSpeech2/evaluation/??"
            audio_pred_dict, acc_res = evalute_ser_result(syn_dir)
            result["emodiff"] = [audio_pred_dict, acc_res]
        else:
            audio_pred_dict, acc_res = None, None
        return acc_res

    def eval_by_pref(self,
                     pref_num=5):
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
                  case_study_example_temp["sad_hap"]
                  ]
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


class Freq_interp_eval():
    def __init__(self):
        pass

    def eval_psdMatch(self):
        pass

    def eval_exampleCase(self):
        """
        compare the sample changing when with and w/o noise guidence
        - with noise guidence
            - 1st half: the harmonic spectrum is changing
            - 2nd half: the other is changing
        - w/o
            - 1st half: overall changing
            - 2nd half: overall changing
        Returns:
        """
        
        pass


class Non_interp_eval():
    def __init__(self):
        pass


    def eval_mos(self, punda=(0.2, 0.8), weight=(0.3, 0.7)):
        for p in punda:
            for w in weight:
                pass

    def eval_emoPref(self, punda=(0.2, 0.8), weight=(0.3, 0.7)):
        for p in punda:
            for w in weight:
                pass


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


if __name__ == '__main__':
    # Non_Interp
    EVAL_NON_INTERP = True

    # temp_Interp
    EVAL_TEMP_INTERP = True
    NEED_SYN_TEMPINTERP = False
    SER_EVAL, PREF_EVAL, TEMP_CASE_EVAL = True, False, False

    # freq_Interp
    EVAL_FREQ_INTERP = False
    NEED_SYN_FREQINTERP = False
    PSD_MATCH_EVAL, FREQ_CASE_EVAL = False, False

    # Non interp
    EVAL_NON_INTERP = False

    temp_eval = InterpEval()
    if EVAL_TEMP_INTERP:
        # sub evaluation
        if SER_EVAL:
            res_acc = temp_eval.eval_by_SER(
                paired_emos=("Angry_Sad", "Sad_Happy", "Angry_Happy"),
                #syn_models=("interpTTS",),
                syn_models=("fastspeech",),
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

    if EVAL_NON_INTERP:
        # input
        p_unda, weight = (0.2, 0.8), (0.3, 0.7)

        non_eval = Non_interp_eval()
        # Case study
        non_eval.eval_emoPref()