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
import torch
from SER import SER
from ..inference_lm import inference_interp
from visualization import show_psd, show_mel
from inference import inference_emodiff_interp, inference_fastspeech

# constant parameter
emo_num_dict = {
    "Angry": 0,
    "Surprise": 1,
    "Sad": 2,
    "Neutral": 3,
    "Happy": 4
}

emo_melstyle_dict = {
    "Angry": ["0019_000508.npy", ""],
    "Surprise": "",
    "Sad": ["0019_001115.npy", ""],
    "Neutral": "",
    "Happy": ""
}

eval_model = {
    "interpTTS": "",
    "emodiff": "",
    "fastspeech": ""
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
    if model_name == "interpTTS":
        checkpoint = "grad_55.pt"
        chk_pt = logs_dir + checkpoint
        syn_txt = "resources/filelists/synthesis1.txt"
        time_steps = 100
        speaker_id = 15
        spk_tensor = torch.LongTensor([speaker_id]).cuda() if not isinstance(speaker_id, type(None)) else None

        chpt_n = checkpoint.split(".")[0].split("_")[1]
        interp_type = "temp"

        out_dir = f"{logs_dir}chpt{chpt_n}_time{time_steps}_interp{interp_type}_spk{speaker_id}_{emo_label1}_{emo_label2}_v3_maskFirstLayers_timeSplit"

        kwargs = {
            "model_name": "gradtts_cross",
            "chk_pt": chk_pt,
            "syn_txt": syn_txt,
            "time_steps": time_steps,
            "spk": spk_tensor,
            "interp_type": interp_type,
        }

        return "", kwargs, out_dir
    elif model_name == "":
        return ""
    elif model_name == "":
        return ""


class Temp_interp_eval:
    def __init__(self,
                 emos,
                 ):
        """
        generate temp interpolation by given below two emotions:
        Ang -> Sad
        Sad -> Hap
        Ang -> Hap
        """
        # Test emotions for Interp
        # Test TTS models (orig, ) for synthesizing
        # SER
        self.paired_emos = ["Angry_Sad", "Sad_Hap", "Angry_Happy"]
        self.ser = SER()
        self.model_names = ["interpTTS", "emodiff", "fastspeech"]

        self.exp_outdir = ""
        self.interp_type = "temp"

        # generate speech from given emo labels and melstyles

        for model_name in self.model_names:
            if model_name == "interpTTS":
                logs_dir_par = "/home/rosen/Project/Speech-Backbones/GradTTS/logs/"
                logs_dir = logs_dir_par + "gradtts_crossSelf_v2/"

                model_config, kwargs, out_dir = get_configs(model_name, logs_dir=logs_dir)
                for pr_emos in self.paired_emos:
                    emo1, emo2 = pr_emos.split("_")
                    emo_label1, emo_label2 = emo_num_dict[emo1], emo_num_dict[emo2]
                    melstyle1, melstyle2 = emo_melstyle_dict[emo1], emo_melstyle_dict[emo2]

                    inference_interp(
                        model_config,
                        emo_label1=emo_label1,
                        melstyle1=melstyle1,
                        emo_label2=emo_label2,
                        melstyle2=melstyle2,
                        **kwargs,
                    )

            elif model_name == "emodiff":
                model_config, kwargs = get_configs(model_name)

                for pr_emos in self.paired_emos:
                    emo1, emo2 = pr_emos.split("_")
                    inference_emodiff_interp(
                        model_config,
                        **kwargs,
                        emo1=emo1,
                        emo2=emo2
                    )

            elif model_name == "fastspeech":
                # combine style tokens
                for pr_emos in self.paired_emos:
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

    def eval_serDetect(self, need_pref_test=True, pref_num=5):
        """
        detect 1st/2nd half emotion by SER
        Returns:

        """
        # SER test for each model
        for model_name in self.model_names:
            audio_gd_dict = get_audio_gd(self.exp_outdir)    # {"a.wav": ("ang", "sad"), ...}
            audio_pred_dict = get_twohalf_emo(self.SER, list(audio_gd_dict.keys()))  # {"a.wav": [("ang", "sad")], ...}
            acc_res = get_acc_res(audio_gd_dict, audio_pred_dict)  # {"ang_sad": (0.8, 0.9), ...}
            print(acc_res)

        # preference Test
        if need_pref_test:
            audio_gd_pref_dict = filter_pref_audio(audio_gd_dict, pref_num=pref_num)
            ## create preference test folder
            audio_gd_pref_dir = create_pref_dir(audio_gd_pref_dict)

            ## DO the Experiment

            ## collect result


    def eval_exampleCase(self):
        """

        Returns:

        """
        audios = [case_study_example_temp["ang_hap"],
                  case_study_example_temp["ang_sad"],
                  case_study_example_temp["sad_hap"]
                  ]
        show_mel(audios)


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


def get_twohalf_emo():
    pass


def get_audio_gd(SER_model, audio_list):
    pass


def get_acc_res():
    pass


def filter_pref_audio():
    pass


def create_pref_dir(audio_gd_pref_dict):
    pass


if __name__ == '__main__':
    TEMP_INTERP = False
    FREW_INTERP = False
    NON_INTERP = False

    if TEMP_INTERP:
        SER_EVAL, PREF_EVAL, CASE_EVAL = False, False, False
        temp_eval = Temp_interp_eval()
        # SER test
        if SER_EVAL:
            temp_eval.eval_serDetect()
        # Pref test

        # Case study
        if CASE_EVAL:
            temp_eval.eval_exampleCase()

    if FREW_INTERP:
        PSD_MATCH_EVAL, CASE_EVAL = False, False

        # PSD match test
        if PSD_MATCH_EVAL:
            temp_eval = Temp_interp_eval()
            temp_eval.eval_serDetect()
        # Case study

    if NON_INTERP:
        # input
        p_unda, weight = (0.2, 0.8), (0.3, 0.7)

        non_eval = Non_interp_eval()
        # Case study
        non_eval.eval_emoPref()