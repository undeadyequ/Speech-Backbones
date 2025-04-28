from typing import Any, Dict, List, Optional, Tuple, Union
import torch

import yaml
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
                               enh_ind=None):
    bone_n, fineAdjNet, input_n = model_n.split("_")
    if bone_n == "stditCross":
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