# if run under test folder
import sys
sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS')

import torch
import yaml
from GradTTS.model.cond_tts import CondGradTTS
from GradTTS.model.cond_diffusion import CondDiffusion
from GradTTS.text.symbols import symbols

def test_CondGradTTS():
    config_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/config/ESD"
    train_config = config_dir + "/train_gradTTS.yaml"
    preprocess_config = config_dir + "/preprocess_gradTTS.yaml"
    model_config = config_dir + "/model_gradTTS.yaml"

    preprocess_config = yaml.load(
        open(preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)

    # preprocess
    add_blank = preprocess_config["feature"]["add_blank"]
    datatype = preprocess_config["datatype"]
    n_spks = int(preprocess_config["feature"]["n_spks"])
    nsymbols = len(symbols) + 1 if add_blank else len(symbols)
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

    ### unet
    unet_type = model_config["unet"]["unet_type"]
    att_type = model_config["unet"]["att_type"]

    ## input value
    text_n = 100
    batch = 2
    n_timesteps = 50
    temperature = 1.0
    stoc = True
    length_scale = 1.0
    text_max_len = 31
    psd_max_len = 31
    mel_max_len = 50
    speaker_n = 10
    style_emb_dim = 786
    speech_len = 70

    emolabel = [[0] * emo_emb_dim, [0] * emo_emb_dim]
    emolabel[0][-1] = 1
    emolabel[0][0] = 1

    inputs_value = {
        "x": torch.randint(0, text_n, (batch, text_max_len)),
        "x_lengths": torch.randint(0, text_max_len, (batch, )),
        "n_timesteps": n_timesteps,
        "temperature": temperature,
        "stoc": stoc,
        "spk": torch.randint(0, speaker_n, (batch, )),
        "length_scale": length_scale,
        #"emo": torch.randn(2, text_n),
        "psd": (
            torch.randn(batch, psd_max_len),
            torch.randn(batch, psd_max_len),
            torch.randn(batch, psd_max_len),
        ),
        "emolabel": torch.tensor(emolabel, dtype=torch.int64),
    }
    """
    x = torch.,
    x_lengths,
    n_timesteps=n_timesteps,
    temperature=temperature,
    stoc=stoc,
    length_scale=length_scale,
    spk=spk,
    emo=emo,
    psd=(pit, eng, dur),
    emolabel = emoLabel

    [str(l) for l in list(range(0, batch))],
    [str(l) for l in list(range(100, 100 + batch))],
    torch.randint(0, speaker_n, (batch,)),
    torch.randint(0, text_n, (batch, text_max_len)),
    torch.randint(0, speaker_n, (batch,)),
    text_max_len,
    torch.randn((batch, mel_max_len, 80)),
    torch.randint(0, mel_max_len, (batch,)),
    mel_max_len,
    torch.randn(batch, text_max_len),
    torch.randn(batch, text_max_len),
    torch.randint(0, 10, (batch, text_max_len)),
    torch.randn(batch, style_emb_dim),
    """
    model = CondGradTTS(nsymbols,
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
                        unet_type,
                        att_type
                        )
    output = model(**inputs_value)

    #
    print(output[0].size(),
          output[1].size(),
          output[2].size())

if __name__ == '__main__':
    test_CondGradTTS()