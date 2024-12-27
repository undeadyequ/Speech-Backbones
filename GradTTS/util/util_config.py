from typing import Any, Dict, List, Optional, Tuple, Union
import torch

import yaml
from GradTTS.text.symbols import symbols
from torch.utils.tensorboard import SummaryWriter
from GradTTS.model.utils import fix_len_compatibility

import numpy as np
from tqdm import tqdm

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



