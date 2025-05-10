from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import yaml
from pooch import retrieve
from GradTTS.text.symbols import symbols
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1

def convert_item2input(item, inference_param=dict(), train_param=dict(), mode="train"):
    if mode == "test":
        x = item['x'].to(torch.long).unsqueeze(0).cuda() # (b, p_l)
        x_len = torch.LongTensor([x.shape[-1]]).cuda()
        spk = item['spk'].to(torch.long).cuda()
        emo_label = item["emo_label"].cuda()
        melstyle = item["melstyle"].cuda()  # (b, d, l)
        melstyle_len = torch.LongTensor([melstyle.shape[-1]]).cuda()

        n_timesteps = inference_param["n_timesteps"]
        temperature = inference_param["temperature"]
        stoc = inference_param["stoc"]
        length_scale =  inference_param["length_scale"]

        model_input = {
            "x": x,
            "x_lengths": x_len,
            "n_timesteps": n_timesteps,
            "temperature": temperature,  #
            "stoc": stoc,
            "spk": spk,
            "length_scale": length_scale,
            "melstyle": melstyle,
            "melstyle_lengths": melstyle_len,
            "emo_label": emo_label,
        }

        # option input depend on model
        if "syl_start" in item.keys():
            syl_start = torch.LongTensor(item['syl_start']).cuda()
            model_input["syl_start"] = syl_start
        return model_input

    elif mode == "train":
        batch = item
        x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda() # (b, p_l)
        y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda() # (b, mel_dim, mel_l)
        spk = batch['spk'].cuda() # (b)
        # emo = batch['emo'].cuda()
        melstyle = batch["melstyle"].cuda()
        melstyle_len = batch["melstyle_lengths"].cuda()  # Use it rather than x_mask
        emo_label = batch["emo_label"].cuda()  # (b, )
        out_size = train_param["out_size"]

        model_input = {
            "x": x,  # (b, p_l)
            "x_lengths": x_lengths,  # (b,)
            "y": y,  # (b, mel_dim, mel_l)
            "y_lengths": y_lengths,  # (b)
            "spk": spk,  # (b)
            "out_size": out_size,
            "melstyle": melstyle,  # (b, ls_dim, ls_l)
            "melstyle_lengths": melstyle_len,
            "emo_label": emo_label  # (b)
        }

        # option input depend on model
        if "syl_start" in item.keys():
            syl_start = torch.LongTensor(item['syl_start']).cuda()
            model_input["syl_start"] = syl_start

        return model_input
    else:
        print("not supported: {}".format(mode))

def convert_to_generator_input(model_n, x, x_lengths, time_steps, temperature, stoc, spk_tensor, length_scale,
                               emo_tensor, melstyle_tensor, melstyle_tensor_lengths, syl_start=None,
                               enh_ind=None, phonemes=None):
    bone_n, fineAdjNet, input_n = model_n.split("_")
    if bone_n == "stditCross":
        if enh_ind is not None:
            if phonemes is not None:
                syn_phoneid_index = convert_phoneid2encPhoneid(enh_ind[0], phonemes)
                ref_phoneid_index = convert_phoneid2encPhoneid(enh_ind[1], phonemes)
                enh_ind = (syn_phoneid_index, ref_phoneid_index)
            else:
                raise IOError("ERROR, Please add phonemes")

        input = {
            "x": x,
            "x_lengths": x_lengths,
            "n_timesteps": time_steps,
            "temperature": temperature, #
            "stoc": stoc,
            "spk": spk_tensor,
            "length_scale": length_scale,
            # "emo": torch.randn(batch, style_emb_dim),
            # "melstyle": torch.randn(batch, style_emb_dim, mel_max_len),# non-vae
            "melstyle": melstyle_tensor,
            "melstyle_lengths": melstyle_tensor_lengths,
            "emo_label": emo_tensor,
            "enh_ind_syn_ref": enh_ind,
        }
        if fineAdjNet == "guideSyl":
            input["syl_start"] = syl_start
    else:
        input = {}
    return input

def convert_enh_syn_ref_index(enh_syn_ref_index):
    """

    Args:
        enh_syn_ref_index: ([0, 1], [0, 1])  # enhancing phones of syn and ref

    Returns:
        converted_enh_syn_ref_index ([5, 8], [5, 8])
    """
    convert_syn_indx = []
    convert_ref_indx = []
    for syn_indx, ref_indx in zip(enh_syn_ref_index[0], enh_syn_ref_index[1]):
        cv_syn_indx = convert_phoneid2encPhoneid(syn_indx)
        convert_syn_indx.append(cv_syn_indx)
        if syn_indx == ref_indx:
            convert_ref_indx.append(cv_syn_indx)
        else:
            cv_ref_indx = convert_phoneid2encPhoneid(ref_indx)
            convert_ref_indx.append(cv_ref_indx)
    return (convert_syn_indx, convert_ref_indx)

def convert_phoneid2encPhoneid(nonEnc_rm_phone_idx, enc_phones):
    """
    
    Convert phoneid of nonEncoded without "" 11 to phoneid of encoded phoneid ()
    Example:
        nonEnc_rm_phone_idx: 2 (AH0)  <- start from 0
        enc_phones: ["", "AH", "", 11, "", "B", "", "AH0", "", "IH, "", 11, "", "AH", "", "G", ""]

    Args:
        phone_idx:
        phones:

    Returns:
        enc_phone_indx: 7
    """
    # find the enhancing phone
    nonEnc_rm_phoneid = [phone for phone in enc_phones if (phone != '' and phone!= 11)]
    enhance_phone = nonEnc_rm_phoneid[nonEnc_rm_phone_idx]
    return nonEnc_rm_phoneid.index(enhance_phone)