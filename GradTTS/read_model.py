import sys
import json
import torch
from pathlib import Path

from GradTTS.text.symbols import symbols
from GradTTS.model import GradTTS, CondGradTTS, CondGradTTSDIT
from typing import Union
sys.path.append('../hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
from GradTTS.model import CondGradTTSDIT3

def get_model1(model_config, model_n="stditCross", chk_pt=None, nsymbols=None):
    if model_n == "stditCross":
        condGradTTSDIT_configs = dict(n_vocab=nsymbols, **model_config["stditCross"])
        generator = CondGradTTSDIT3(**condGradTTSDIT_configs)
    else:
        print("{} on building!")
    load_model_state(chk_pt, generator)
    generator.cuda().eval()
    return generator

def get_model(configs, model="gradtts_lm", chk_pt=None):
    """
    Get Model
    """
    preprocess_config, model_config, train_config = configs

    # parameter
    add_blank = preprocess_config["feature"]["add_blank"]
    nsymbols = len(symbols) + 1 if add_blank else len(symbols)
    n_spks = int(preprocess_config["feature"]["n_spks"])
    n_feats = int(preprocess_config["feature"]["n_feats"])

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
    beta_min = float(model_config["decoder"]["beta_min"])
    beta_max = float(model_config["decoder"]["beta_max"])
    pe_scale = int(model_config["decoder"]["pe_scale"])
    stoc = model_config["decoder"]["stoc"]
    temperature = float(model_config["decoder"]["temperature"])
    n_timesteps = int(model_config["decoder"]["n_timesteps"])
    sample_channel_n = int(model_config["decoder"]["sample_channel_n"])
    melstyle_n = int(model_config["decoder"]["melstyle_n"])
    psd_n = int(model_config["decoder"]["psd_n"])

    ### unet
    unet_type = model_config["unet"]["unet_type"]
    att_type = model_config["unet"]["att_type"]
    att_dim = model_config["unet"]["att_dim"]
    heads = model_config["unet"]["heads"]
    p_uncond = model_config["unet"]["p_uncond"]

    # dit_mocha config
    dit_mocha = model_config["dit_mocha"]
    stdit = model_config["stdit"]
    stditMocha = model_config["stditMocha"]

    guided_attn = model_config["loss"]["guided_attn"]
    diff_model = model_config["diff_model"]
    ref_encoder = model_config["ref_encoder"]
    ref_embedder = model_config["ref_embedder"]
    # vqvae config
    vqvae = model_config["vqvae"]

    if model == "gradtts_cross" or model == "gradtts":
        generator = CondGradTTS(nsymbols,
                            n_spks,
                            spk_emb_dim,
                            emo_emb_dim,
                            n_enc_channels,
                            filter_channels,
                            filter_channels_dp,
                            n_heads,
                            n_enc_layers,
                            enc_kernel,
                            enc_dropout,
                            window_size,
                            n_feats,
                            dec_dim,
                            sample_channel_n,
                            beta_min,
                            beta_max,
                            pe_scale,
                            unet_type,
                            att_type,
                            att_dim,
                            heads,
                            p_uncond)
        load_model_state(chk_pt, generator)
        generator.cuda().eval()
    elif "STDit" in model:
        generator = CondGradTTSDIT(
            nsymbols,
            n_spks,
            spk_emb_dim,
            emo_emb_dim,
            n_enc_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_enc_layers,
            enc_kernel,
            enc_dropout,
            window_size,
            n_feats,
            dec_dim,
            sample_channel_n,
            beta_min,
            beta_max,
            pe_scale,
            unet_type,
            att_type,
            att_dim,
            heads,
            p_uncond,
            psd_n,
            melstyle_n,  # 768
            diff_model=diff_model,
            ref_encoder=ref_encoder,
            guided_attn=guided_attn,
            dit_mocha_config=dit_mocha,
            stdit_config=stdit,
            stditMocha_config=stditMocha,
            vqvae=vqvae
        )
    else:
        pass

    load_model_state(chk_pt, generator)
    generator.cuda().eval()
    return generator

import yaml


def read_model_configs(config_dir, ptm_configs=tuple()):
    prep, model, train = ptm_configs
    preprocess_config = yaml.load(
        open(config_dir + "/" + prep, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(
        config_dir + "/" + model, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        config_dir + "/" + train, "r"), Loader=yaml.FullLoader)

    return preprocess_config, model_config, train_config


def modify_submodel_config(ptm_configs, prep_mody=None, model_mody=None, train_mody=None):
    prep, model, train = ptm_configs
    return adjust_config(prep, prep_mody), adjust_config(model, model_mody), adjust_config(train, train_mody)


def adjust_config(origin_dict, mody=None):
    adjusted_config = origin_dict.copy()
    if mody is None:
        return origin_dict
    else:
        for k, v in mody.items():
            if isinstance(v, dict):
                for k1, v in v.items():
                    if k1 in adjusted_config[k].keys():
                        adjusted_config[k][k1] = v
                    else:
                        IOError(f"{k1} is not in given p/t/m_config")
            else:
                if k in adjusted_config.keys():
                    adjusted_config[k] = v
                else:
                    IOError(f"{k} is not in given p/t/m_config")
        return adjusted_config


def load_model_state(checkpoint: Union[str, Path], model: torch.nn.Module, ngpu=1):
    ckpt_states = torch.load(
        checkpoint,
        map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu",
    )

    if "model_state_dict" in ckpt_states:
        model.load_state_dict(
            ckpt_states["model_state_dict"])
    else:  # temp use
        states = torch.load(
            checkpoint,
            map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu",
        )
        model.load_state_dict(states)


def get_vocoder():
    HIFIGAN_CONFIG = '/home/rosen/Project/Speech-Backbones/GradTTS/hifi-gan/checkpts/hifigan-config.json'  # ./checkpts/config.json
    HIFIGAN_CHECKPT = '/home/rosen/Project/Speech-Backbones/GradTTS/hifi-gan/checkpts/hifigan.pt'
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    return vocoder