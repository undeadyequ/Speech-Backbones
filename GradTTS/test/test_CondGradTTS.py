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
    style_emb_dim = 768
    mel_emb_dim = 80
    speech_len = 70

    emolabel = [[0] * emo_emb_dim, [0] * emo_emb_dim]
    emolabel[0][-1] = 1
    emolabel[0][0] = 1

    """
    "psd": (
        torch.randn(batch, psd_max_len),
        torch.randn(batch, psd_max_len),
        torch.randn(batch, psd_max_len),
    ),
    """

    TEST_REVERSE = False
    TEST_REVERSE_INTERP_TEMP = False
    TEST_REVERSE_INTERP_FREQ = True
    TEST_COMPUTE_LOSS = False

    # test condition
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
                        sample_channel_n,
                        beta_min,
                        beta_max,
                        pe_scale,
                        unet_type,
                        att_type,
                        att_dim,
                        heads,
                        p_uncond
                        )

    if TEST_REVERSE:
        inputs_value = {
            "x": torch.randint(0, text_n, (batch, text_max_len)),
            "x_lengths": torch.randint(0, text_max_len, (batch,)),
            "n_timesteps": n_timesteps,
            "temperature": temperature,
            "stoc": stoc,
            "spk": torch.randint(0, speaker_n, (batch,)),
            "length_scale": length_scale,
            "psd": (None, None, None),
            #"emo": torch.randn(batch, style_emb_dim),
            "melstyle": torch.randn(batch, style_emb_dim, mel_max_len),
            "emo_label": torch.tensor(emolabel, dtype=torch.int64)
        }
        output = model(**inputs_value)

        # encoder_outputs, decoder_outputs, attn
        print("print encoder_outputs: {}, decoder_outputs: {}, attn: {}".format(
            output[0].size(),
            output[1].size(),
            output[2].size())
        )
    if TEST_REVERSE_INTERP_TEMP:
        emolabel1 = [[0] * emo_emb_dim]
        emolabel1[0][0] = 1
        emolabel2 = [[0] * emo_emb_dim]
        emolabel2[0][1] = 1
        batch = 1
        mel_max_len = 2
        inputs_value = {
            "x": torch.randint(0, text_n, (batch, text_max_len)),
            "x_lengths": torch.randint(0, text_max_len, (batch,)),
            "n_timesteps": n_timesteps,
            "temperature": temperature,
            "stoc": stoc,
            "length_scale": length_scale,
            "spk": torch.randint(0, speaker_n, (batch,)),
            "melstyle1": torch.randn(batch, style_emb_dim, mel_max_len),
            "melstyle2": torch.randn(batch, style_emb_dim, mel_max_len),
            "emo_label1": torch.tensor(emolabel1, dtype=torch.int64),
            "emo_label2": torch.tensor(emolabel2, dtype=torch.int64),
            "interp_type": "temp"
        }

        y_enc, y_dec, attn = model.reverse_diffusion_interp(
            **inputs_value
        )

        # encoder_outputs, decoder_outputs, attn
        print("print encoder_outputs: {}, decoder_outputs: {}, attn: {}".format(
            y_enc.size(),
            y_dec.size(),
            attn.size())
        )

    if TEST_REVERSE_INTERP_FREQ:
        pass

    if TEST_COMPUTE_LOSS:
        inputs_value_train = {
            "x": torch.randint(0, text_n, (batch, text_max_len)),
            "x_lengths": torch.randint(0, text_max_len, (batch,)),
            "y": torch.randn(batch, mel_emb_dim, mel_max_len),
            "y_lengths": torch.randint(0, mel_max_len, (batch,)),
            "spk": torch.randint(0, speaker_n, (batch,)),
            "out_size": 172,
            #"psd": (None, None, None),
            "melstyle": torch.randn(batch, style_emb_dim, mel_max_len),
            "emo_label": torch.tensor(emolabel, dtype=torch.float32),
            #"emo_label": None,
        }
        for i in range(15):
            dur_loss, prior_loss, diff_loss = model.compute_loss(**inputs_value_train)
            print(dur_loss, prior_loss, diff_loss)

if __name__ == '__main__':
    test_CondGradTTS()