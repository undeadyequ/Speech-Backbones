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
        phoneme = item['phoneme']
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
            "phoneme": phoneme
        }

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

        return model_input
    else:
        print("not supported: {}".format(mode))


def convert_to_generator_input(model_n, x, x_lengths, time_steps, temperature, stoc, spk_tensor, length_scale,
                            emo_tensor, melstyle_tensor, melstyle_tensor_lengths):
    if "stditCross" in model_n:
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
            "emo_label": emo_tensor
        }
    return input

def convert_originConfig_to_styleEnhanceConfig(train_config, preprocess_config, model_config):
    log_dir = train_config["path"]["log_dir"]

    # set seed
    torch.manual_seed(train_config["seed"])
    np.random.seed(train_config["seed"])
    # logger
    logger = SummaryWriter(log_dir=log_dir)
    # preprocess
    add_blank = preprocess_config["feature"]["add_blank"]
    sample_rate = preprocess_config["feature"]["sample_rate"]
    nsymbols = len(symbols) + 1 if add_blank else len(symbols)
    out_size = fix_len_compatibility(2 * sample_rate // 256)
    preprocess_dir = preprocess_config["path"]["preprocessed_path"]
    train_filelist_path = preprocess_dir + "/train.txt"
    valid_filelist_path = preprocess_dir + "/val.txt"
    meta_json_path = preprocess_dir + "/metadata_new.json"

    cmudict_path = preprocess_config["path"]["cmudict_path"]
    n_fft = int(preprocess_config["feature"]["n_fft"])
    n_feats = int(preprocess_config["feature"]["n_feats"])
    sample_rate = int(preprocess_config["feature"]["sample_rate"])
    hop_length = int(preprocess_config["feature"]["hop_length"])
    win_length = int(preprocess_config["feature"]["win_length"])
    f_min = int(preprocess_config["feature"]["f_min"])
    f_max = int(preprocess_config["feature"]["f_max"])
    n_spks = int(preprocess_config["feature"]["n_spks"])

    datatype = preprocess_config["datatype"]

    # model
    ## Encoder
    spk_emb_dim = int(model_config["spk_emb_dim"])
    emo_emb_dim = int(model_config["emo_emb_dim"])
    n_enc_channels = int(model_config["encoder"]["n_enc_channels"])
    filter_channels = int(model_config["encoder"]["filter_channels"])
    filter_channels_dp = int(model_config["encoder"]["filter_channels_dp"])
    n_heads = int(model_config["encoder"]["n_heads"])
    n_enc_layers = int(model_config["encoder"]["n_enc_layers"])
    enc_kernel = int(model_config["encoder"]["enc_kernel"])
    enc_dropout = float(model_config["encoder"]["enc_dropout"])
    window_size = int(model_config["encoder"]["window_size"])
    length_scale = float(model_config["encoder"]["length_scale"])

    ## Decoder
    dec_dim = int(model_config["decoder"]["dec_dim"])
    sample_channel_n = int(model_config["decoder"]["sample_channel_n"])
    beta_min = float(model_config["decoder"]["beta_min"])
    beta_max = float(model_config["decoder"]["beta_max"])
    pe_scale = int(model_config["decoder"]["pe_scale"])
    stoc = model_config["decoder"]["stoc"]
    temperature = float(model_config["decoder"]["temperature"])
    n_timesteps = int(model_config["decoder"]["n_timesteps"])

    ### unet
    unet_type = model_config["unet"]["unet_type"]
    att_type = model_config["unet"]["att_type"]
    att_dim = model_config["unet"]["att_dim"]
    heads = model_config["unet"]["heads"]
    p_uncond = model_config["unet"]["p_uncond"]

    # train
    batch_size = int(train_config["batch_size"])
    learning_rate = float(train_config["learning_rate"])
    n_epochs = int(train_config["n_epochs"])
    resume_epoch = int(train_config["resume_epoch"])
    save_every = int(train_config["save_every"])
    ckpt = f"{log_dir}models/grad_{resume_epoch}.pt"

    # dit_mocha config
    dit_mocha = model_config["dit_mocha"]
    guided_attn = model_config["loss"]["guided_attn"]

    # vqvae config
    vqvae = model_config["vqvae"]


    emodiff_config = {
        "n_vocab": nsymbols,
        "n_feats": n_feats,
        "n_enc_channels": n_enc_channels,
        "filter_channels": filter_channels,
        "filter_channels_dp": filter_channels_dp,
        "n_heads": n_heads,
        "n_enc_layers": n_enc_layers,
        "enc_kernel": enc_kernel,
        "enc_dropout": enc_dropout,
        "window_size": window_size,
        "dec_dim": dec_dim,
        "sample_channel_n": sample_channel_n,
        "n_spks": n_spks,
        "spk_emb_dim": spk_emb_dim,
        "pe_scale": pe_scale,
        "emo_emb_dim": emo_emb_dim,
        "att_type": att_type,
        "att_dim": att_dim,
        "heads": heads,
        "p_uncond": p_uncond,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "conv1d_dim": conv1d_dim,
        "layer_n": layer_n,
        "dropout_rate": dropout_rate,
        "kernel": kernel,
        "padding": padding,
        "beta_min": beta_min,
        "beta_max": beta_max
    }

    return emodiff_config



