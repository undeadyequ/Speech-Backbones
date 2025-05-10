import sys, os
from GradTTS.preprocessor.preprocessor_v2 import PreprocessorExtract
import time

def mta_speech(out_sub_dir, tg_dir):
    ## Conduct alignment by "python mfa_dir.py ~/data/test_data/ english_mfa ~/data/tg/" under aligner conda enviroment
    print("####################")
    print("Please conduct alignment by: 1> conda activate aligner\n")
    print("2> python mfa_dir.py {} english_mfa {}".format(
        out_sub_dir, tg_dir
    ))
    print(time.sleep(5))
    print("####################")
    sys.exit()


def extract_psd(preprocess_config, model_name1, model_name2, out_dir):
    """
    # Extract PSD given wav/tgt and save json
    Args:
        configs1:
    Returns:

    """
    prosody_dict = dict()
    preprocessor = PreprocessorExtract(preprocess_config)
    # Do MFA
    for i, model_n in enumerate(["reference", model_name1, model_name2]):
        ## Create output dir
        out_speech_dir = os.path.join(out_dir, model_n)  # audio_txt folder
        out_mfa_dir = os.path.join(out_dir, model_n + "_mfa")  # textgrid folder
        if not os.path.isdir(out_mfa_dir):
            mta_speech(out_speech_dir, out_mfa_dir)
        else:
            print("Escape mfa!")

    # Do PSD extraction
    for i, model_n in enumerate(["reference", model_name1, model_name2]):
        out_speech_dir = os.path.join(out_dir, model_n)  # audio_txt folder
        out_mfa_dir = os.path.join(out_dir, model_n + "_mfa")  # textgrid folder
        out_psd_dir = os.path.join(out_dir, model_n + "_psd")  # psd folder
        if not os.path.isdir(out_mfa_dir):
            mta_speech(out_speech_dir, out_mfa_dir)

        speech_list = [speech for speech in os.listdir(out_speech_dir) if speech.endswith(".wav")]
        print("Do psd extraction!")
        for speech in speech_list:
            speech_f = os.path.join(out_speech_dir, speech)

            spk, emo_id = speech.split("_")[:2]
            ######## CHECK1
            #if speech != "spk19_Surprise_txt0.wav":   # high pitch at last word!
            #    continue
            if spk not in prosody_dict.keys():
                prosody_dict[spk] = dict()
            if emo_id not in prosody_dict[spk].keys():
                prosody_dict[spk][emo_id] = dict()
            if model_n not in prosody_dict[spk][emo_id].keys():
                prosody_dict[spk][emo_id][model_n] = {
                    "phonemes": [],
                    "pitch": [],
                    "energy": [],
                    "duration": [],
                    "speechid": []
                }
            tg_path = os.path.join(out_mfa_dir, "{}.TextGrid".format(os.path.basename(speech_f).split(".")[0]))
            try:
                phonemes, pitch, energy, mels, duration = preprocessor.extract_pitch_energy_mel(speech_f,
                                                                                                tg_path=tg_path,
                                                                                                out_dir=out_psd_dir,
                                                                                                average_phoneme=True,
                                                                                                save_npy=True)
            except IOError:
                print("{} is failed to extract psd".format(speech_f))

            prosody_dict[spk][emo_id][model_n]["phonemes"].append(phonemes.split(" "))
            prosody_dict[spk][emo_id][model_n]["pitch"].append(pitch.tolist())
            prosody_dict[spk][emo_id][model_n]["energy"].append(energy.tolist())
            prosody_dict[spk][emo_id][model_n]["duration"].append(duration)
            prosody_dict[spk][emo_id][model_n]["speechid"].append(speech.split(".")[0])
    return prosody_dict
