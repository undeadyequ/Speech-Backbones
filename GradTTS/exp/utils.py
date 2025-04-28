from pymcd.mcd import Calculate_MCD
import librosa
import soundfile as sf


# instance of MCD class
# three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics

mcd_toolbox = Calculate_MCD(MCD_mode="MCD-DTW")

# two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively
#mcd_value = mcd_toolbox.calculate_mcd("001.wav", "002.wav")import librosa
from pymcd.mcd import Calculate_MCD
import torch
import tgt
import numpy as np
import os
from GradTTS.const_param import textgrid_dir
import sys, os, yaml, json
import numpy as np
from pathlib import Path
import torch
from GradTTS.const_param import emo_melstyleSpk_dict, config_dir, logs_dir, melstyle_dir, wav_dir, wav_dict, emo_num_dict, logs_dir_par, psd_quants_dir
import shutil
from GradTTS.utils import parse_filelist, intersperse, get_emo_label

##################### Operate file ##################
def convert_style2json(style_id):
    pass

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
    """
    he was still in the forest!
    he was still in the forest!
    he was still in the forest!
    """
    with open(synText_f, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    return texts


def get_synStyle_from_file(synStyle_f,
                           split_char="|",
                           melstyle_type="codec",
                           wav_dir=wav_dir,
                           psd_quants_dir=psd_quants_dir,
                           melstyle_dir=melstyle_dir):
    """
    get style related features from file with below format

    0019_001103|0019|{HH IY1 W AH0 Z S T IH1 L IH0 N TH IY0 F AO1 R AH0 S T}|he was still in the forest!|Sad
    psd_quants_dir: FACodec
    melstyle_dir:   wav2vec2
    """
    styles = []
    syn_styles = parse_filelist(synStyle_f, split_char=split_char)  # emotion changed

    if len(syn_styles[0]) == 5:
        for speechid, spk, phoneme, txt, emo in syn_styles:
            spk = int(spk[2:])
            wav_n = os.path.join(wav_dir, speechid + ".wav")
            emo_tensor = get_emo_label(emo, emo_num_dict, emo_type="index")
            spk_tensor = torch.LongTensor([spk]).cuda()
            if melstyle_type == "codec":
                melstyle_tensor = torch.from_numpy(np.load(psd_quants_dir + "/" + speechid + ".npy")).cuda()
            else:
                melstyle_tensor = torch.from_numpy(np.load(melstyle_dir + "/" + speechid + ".npy")).cuda()
                melstyle_tensor = melstyle_tensor.unsqueeze(0).transpose(1, 2)
            styles.append((emo, spk, wav_n, txt, emo_tensor, melstyle_tensor, spk_tensor))
    elif len(syn_styles[0]) == 6:
        for speechid, spk, phoneme, txt, syl_start, emo in syn_styles:
            spk = int(spk[2:])
            wav_n = os.path.join(wav_dir, speechid + ".wav")
            emo_tensor = get_emo_label(emo, emo_num_dict, emo_type="index")
            spk_tensor = torch.LongTensor([spk]).cuda()
            if melstyle_type == "codec":
                melstyle_tensor = torch.from_numpy(np.load(psd_quants_dir + "/" + speechid + ".npy")).cuda()
            else:
                melstyle_tensor = torch.from_numpy(np.load(melstyle_dir + "/" + speechid + ".npy")).cuda()
                melstyle_tensor = melstyle_tensor.unsqueeze(0).transpose(1, 2)
            syl_start = torch.tensor([int(i) for i in syl_start.split(",")], dtype=torch.long).unsqueeze(0)
            styles.append((emo, spk, wav_n, txt, emo_tensor, melstyle_tensor, spk_tensor, syl_start))
    return styles


def fine_adjust_configs(bone_n, bone_option, bone2configPath, config_dir):
    """
    fine-adjust model/train configs of bone_n by bone_option
    Returns:

    """
    log_dir_base = "/home/rosen/Project/Speech-Backbones/GradTTS/logs/{}/"
    preprocess, model_config, train_config = bone2configPath[bone_n]
    preprocess_config = yaml.load(
        open(config_dir + "/" + preprocess, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
        config_dir + "/" + model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        config_dir + "/" + train_config, "r"), Loader=yaml.FullLoader)
    if bone_n == "stditCross":
        if bone_option == "noguide":
            model_config["stditCross"]["guide_loss"] = False
            train_config["path"]["log_dir"] = log_dir_base.format("stditCross_base_codec")
        elif bone_option == "guideframe":
            model_config["stditCross"]["guide_loss"] = True
            model_config["stditCross"]["decoder_config"]["stdit_config"]["phoneme_RoPE"] = True
            train_config["path"]["log_dir"] = log_dir_base.format("stditCross_guideLoss_codec")
        elif bone_option == "guidephone":
            model_config["stditCross"]["guide_loss"] = True
            model_config["stditCross"]["decoder_config"]["stdit_config"]["phoneme_RoPE"] = True
            train_config["path"]["log_dir"] = log_dir_base.format("stditCross_guideLoss_codec_phoneRope")
        elif bone_option == "guideSyl":
            model_config["stditCross"]["guide_loss"] = True
            model_config["stditCross"]["decoder_config"]["stdit_config"]["phoneme_RoPE"] = True
            train_config["path"]["log_dir"] = log_dir_base.format("stditCross_guideLoss_codec_sylRope")
    return preprocess_config, model_config, train_config

def copy_ref_speech(src_wavs, src_txts, dst_wavs, dst_txts):
    for src_wav, src_txt, dst_wav, dst_txt in zip(src_wavs, src_txts, dst_wavs, dst_txts):
        shutil.copyfile(src_wav, dst_wav)
        with open(dst_txt, "w") as file1:
            # Writing data to a file
            file1.write(src_txt)

def convert_vis_psd_json(prosody_dict_json, show_txt_id=0):
    """
    {"emo1": {"model1": {"psd1/phone": []}}} -> {"emo1": {"model1": list(pitch_len)}}}
    """
    pitch_dict_for_vis = dict()  # for_vis: {"emo1": {"model1": list(p_len)}}}
    for emo, m_psd in prosody_dict_json.items():
        pitch_dict_for_vis[emo] = {}
        for model, psd in m_psd.items():
            if model not in pitch_dict_for_vis[emo].keys():
                pitch_dict_for_vis[emo][model] = psd["pitch"][show_txt_id]

    energy_dict_for_vis = dict()  # {"emo1": {"model1": list(p_len)}}}
    for emo, m_psd in prosody_dict_json.items():
        energy_dict_for_vis[emo] = {}
        for model, psd in m_psd.items():
            if model not in energy_dict_for_vis[emo].keys():
                energy_dict_for_vis[emo][model] = psd["energy"][show_txt_id]

    return pitch_dict_for_vis, energy_dict_for_vis

def convert_attn_json(prosody_dict_json, out_dir, show_t=0, show_b=0, show_h=0, show_txt=0):
    """
    For visualizing crossAttn
    - auxiliary line given mfa duration ?

    """
    batch_n = 0
    crossAttn_dict_for_vis = dict()  # {"model1": {"emo1": np.array([syn_frames, ref_frames])}}}  <- choose t and h from attn_map_dict
    crossAttn_dict_for_vis_aux = dict()  # {"model1": {"emo1": ([pdurs_nums], [p_nums])}}}
    print("4. Vis attention map given attn file")

    for emo, m_psd in prosody_dict_json.items():
        for model, psd in m_psd.items():
            if model != "reference":
                crossAttn_dir = os.path.join(out_dir, model + "_attn/")
                crossAttn_f = crossAttn_dir + prosody_dict_json[emo][model]["speechid"][show_txt] + ".npy"
                if model not in crossAttn_dict_for_vis.keys():
                    crossAttn_dict_for_vis[model] = {}
                if emo not in crossAttn_dict_for_vis.keys():
                        crossAttn_dict_for_vis[model][emo] = {}

                # attn, syn_durs, syn_phones, ref_durs, ref_phones
                crossAttn_dict_for_vis[model][emo] = (
                    np.load(crossAttn_f, allow_pickle=True)[show_t, show_b, batch_n, show_h, ...], # [time, block_n, batch, head, t_t, t_s]
                    prosody_dict_json[emo][model]["k_dur"][show_txt],
                    prosody_dict_json[emo][model]["phonemes"][show_txt],
                    prosody_dict_json[emo][model]["q_dur"][show_txt],
                    prosody_dict_json[emo][model]["phonemes"][show_txt],
                )
    return crossAttn_dict_for_vis


def convert_attn_json_bk(prosody_dict_json, out_dir, show_t=0, show_b=0, show_h=0, show_txt=0):
    """
    For visualizing crossAttn
    - auxiliary line given mfa duration ?

    """
    batch_n = 0
    crossAttn_dict_for_vis = dict()  # {"model1": {"emo1": np.array([syn_frames, ref_frames])}}}  <- choose t and h from attn_map_dict
    crossAttn_dict_for_vis_aux = dict()  # {"model1": {"emo1": ([pdurs_nums], [p_nums])}}}
    print("4. Vis attention map given attn file")

    for emo, m_psd in prosody_dict_json.items():
        for model, psd in m_psd.items():
            if model != "reference":
                crossAttn_dir = os.path.join(out_dir, model + "_attn/")
                crossAttn_f = crossAttn_dir + prosody_dict_json[emo][model]["speechid"][show_txt] + ".npy"
                if model not in crossAttn_dict_for_vis.keys():
                    crossAttn_dict_for_vis[model] = {}
                if emo not in crossAttn_dict_for_vis.keys():
                        crossAttn_dict_for_vis[model][emo] = {}

                # attn, syn_durs, syn_phones, ref_durs, ref_phones
                crossAttn_dict_for_vis[model][emo] = (
                    np.load(crossAttn_f, allow_pickle=True)[show_t, show_b, batch_n, show_h, ...], # [time, block_n, batch, head, t_t, t_s]
                    prosody_dict_json[emo][model]["duration"][show_txt],
                    prosody_dict_json[emo][model]["phonemes"][show_txt],
                    prosody_dict_json[emo]["reference"]["duration"][show_txt],
                    prosody_dict_json[emo]["reference"]["phonemes"][show_txt],
                )
    return crossAttn_dict_for_vis


def get_pitch_match_score(pitch1, pitch2):
    pitch_score = 0
    return pitch_score


def get_pitch_match_score(pitch1, pitch2):
    pitch_score = 0
    return pitch_score


def get_ref_pRange(p_start=5, p_end=8, ref_id="0019_000403"):
    """
    # S(5) T(6) IH1(7) L(8)
    get range of specific phoneme on frames by given phoneme index of reference speech id
    """
    # get the time/frame range of given ref text (Know from ref id)
    speaker = ref_id.split("_")[0]
    tg_path = os.path.join(
        textgrid_dir, speaker, "{}.TextGrid".format(ref_id)
    )
    textgrid = tgt.io.read_textgrid(tg_path)
    # duration = frame number
    phone, duration, start, end = get_alignment(
        textgrid.get_tier_by_name("phones")
    )
    #pIndex = get_pIndex_from_wIndx(ref_wIndx) # ?? word ??
    ref_nRange = (sum(duration[:p_start]), sum(duration[:p_end+1]))
    #frame_n = sum(duration)
    # check if ref_nRange match the spefied frame (still) ??
    return ref_nRange


def get_tgt_pRange():
    pass

def get_pIndex_from_wIndx(ref_wIndx):
    pass


def get_alignment(tier):
    sil_phones = ["sil", "sp", "spn"]

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    sampling_rate = 16000
    hop_length = 256

    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            # For silent phones
            phones.append(p)

        durations.append(
            int(
                np.round(e * sampling_rate / hop_length)
                - np.round(s * sampling_rate / hop_length)
            )
        )

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    return phones, durations, start_time, end_time

def compute_phoneme_mcd(wav1,
                        wav2,
                        p1_interv=(0, 1),
                        p2_interv=(0, 1),
                        ):
    # read wav1, wav2
    y1, sr1 = librosa.load(wav1, sr=None)
    y2, sr2 = librosa.load(wav2, sr=None)
    # split
    y1_split = y1[p1_interv[0] * sr1 : p1_interv[1]]
    y2_split = y2[p2_interv[0] * sr2 : p2_interv[1]]

    # write to file
    wav1_split = wav1.split(".")[0] + "_p{}_{}.wav".format(
        p1_interv[0], p1_interv[1])
    wav2_split = wav2.split(".")[0] + "_p{}_{}.wav".format(
        p2_interv[0], p2_interv[1])
    sf.write(wav1_split, y1_split, sr1)
    sf.write(wav2_split, y2_split, sr2)
    # mcd
    mcd_toolbox.calculate_mcd(wav1_split, wav2_split)

def convert_frameNum2second(frame_num, sr=16000, chunk_len=8000):
    return frame_num * chunk_len / sr


############ vis related ############
def convert_xydur_xybox(x_dur, y_dur):
    if len(x_dur)!=len(y_dur):
        IOError("xdur and ydur should be same length: {}, {}".format(len(x_dur), len(y_dur)))
    xywh_list = []
    for i, (xd, yd) in enumerate(zip(x_dur, y_dur)):
        if i < len(x_dur)-1:
            x, y = xd, yd
            w, h = x_dur[i+1] - xd, y_dur[i+1] - yd
            xywh_list.append((x, y, w, h))
    return xywh_list


if __name__ == '__main__':
    speech_dir = ("/home/rosen/Project/Speech-Backbones/GradTTS/logs/interpEmoTTS_frame2binAttn_noJoint/"
                  "chpt400_time50_spk19_emoAngry")
    non_p2p = speech_dir + "/no_p2p_sample_v1_0.wav"
    p2p = speech_dir + "/p2p_sample_v1_0.wav"

    origin_dir = "/home/rosen/Project/FastSpeech2/ESD/16k_wav"
    origin = origin_dir + "/0019_000403.wav"

    mcd_origin_nonP2P = compute_phoneme_mcd(
         non_p2p,
        origin,
        p1_interv=(0, 1),
        p2_interv=(0, 1)
    )

    mcd_origin_p2p = compute_phoneme_mcd(
        p2p,
        origin,
        p1_interv=(0, 1),
        p2_interv=(9, 1)
    )
    print("mcd betwenn origin and nonp2p is: {}".format(mcd_origin_nonP2P))
    print("mcd betwenn origin and p2p is: {}".format(mcd_origin_p2p))


