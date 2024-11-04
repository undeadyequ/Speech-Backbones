import sys
import json
import torch

from GradTTS.text.symbols import symbols
from GradTTS.model import GradTTS, CondGradTTS, CondGradTTSLDM

sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN


def get_model(configs, model="gradtts_lm"):
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

    ### unet
    unet_type = model_config["unet"]["unet_type"]
    att_type = model_config["unet"]["att_type"]
    att_dim = model_config["unet"]["att_dim"]
    heads = model_config["unet"]["heads"]
    p_uncond = model_config["unet"]["p_uncond"]

    """
    if model == "gradtts_lm":
        return CondGradTTSLDM(
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
            beta_min,
            beta_max,
            pe_scale,
            att_type)
    """
    if model == "gradtts_cross":
        return CondGradTTS(nsymbols,
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

def get_vocoder():
    HIFIGAN_CONFIG = './checkpts/hifigan-config.json'  # ./checkpts/config.json
    HIFIGAN_CHECKPT = './checkpts/hifigan.pt'
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    return vocoder