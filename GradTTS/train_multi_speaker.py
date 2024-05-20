# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
import sys
sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS/model')

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#import params
from model import GradTTS, CondGradTTS
from data import TextMelSpeakerDataset, TextMelSpeakerBatchCollate, TextMelSpeakerEmoDataset, TextMelSpeakerEmoBatchCollate
from fastext_dataset import FastspeechDataset
from utils import plot_tensor, save_plot
from text.symbols import symbols
import yaml
from model.utils import fix_len_compatibility
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
from scipy.io.wavfile import write
import json


def train_process_cond(configs):
    preprocess_config, model_config, train_config = configs
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
    ckpt = f"{log_dir}grad_{resume_epoch}.pt"

    # dataset
    train_dataset = TextMelSpeakerEmoDataset(train_filelist_path,
                                             meta_json_path,
                                             cmudict_path,
                                             preprocess_dir,
                                             add_blank,
                                             n_fft, n_feats,
                                             sample_rate, hop_length,
                                             win_length, f_min, f_max,
                                             datatype=datatype
                                             )
    test_dataset = TextMelSpeakerEmoDataset(valid_filelist_path,
                                            meta_json_path,
                                            cmudict_path,
                                            preprocess_dir,
                                            add_blank,
                                            n_fft, n_feats,
                                            sample_rate, hop_length,
                                            win_length, f_min, f_max,
                                            datatype=datatype
                                            )

    batch_collate = TextMelSpeakerEmoBatchCollate()

    loader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        collate_fn=batch_collate,
                        drop_last=True,
                        num_workers=8,
                        shuffle=True)

    # get test_batch size
    test_batch = test_dataset.sample_test_batch(size=3)

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
                        p_uncond,
                        ).cuda()
    print(model)
    """
    print('Number of encoder parameters = %.2fm' % (model.encoder.nparams / 1e6))
    print('Number of decoder parameters = %.2fm' % (model.decoder.nparams / 1e6))
    print('Initializing optimizer...')
    """

    if resume_epoch > 1:
        resume(ckpt, model, 1)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print('Logging test batch...')
    for item in test_batch:
        mel, spk = item['y'], item['spk']
        i = int(spk.cpu())
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')

    print('Start training...')
    iteration = 0
    show_img_per_epoch = 5.0
    # create vocoder
    vocoder = get_vocoder()
    for epoch in range(resume_epoch + 1, n_epochs + 1):
        model.eval()
        print('Synthesis...')
        with torch.no_grad():
            for item in test_batch:
                x = item['x'].to(torch.long).unsqueeze(0).cuda()
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                spk = item['spk'].to(torch.long).cuda()
                emo, pit, eng, dur, emo_label = None, None, None, None, None
                if "emo" in item.keys():
                    emo = item['emo'].cuda()
                if "pit" in item.keys() and "eng" in item.keys() and "dur" in item.keys():
                    pit = item["pit"].cuda()
                    eng = item["eng"].cuda()
                    dur = item["dur"].cuda()
                if "emo_label" in item.keys():
                    emo_label = item["emo_label"].cuda()
                if "melstyle" in item.keys():
                    melstyle = item["melstyle"].cuda()

                i = int(spk.cpu())
                y_enc, y_dec, attn = model(x,
                                           x_lengths,
                                           n_timesteps=n_timesteps,
                                           temperature=temperature,
                                           stoc=stoc,
                                           length_scale=length_scale,
                                           spk=spk,
                                           #emo=emo,
                                           #psd=(pit, eng, dur),
                                           emo_label=emo_label,
                                           melstyle=melstyle
                                           )

                logger.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/alignment',
                                 plot_tensor(attn.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                # show image and audio (show initiate image to check earlierly)
                if epoch % show_img_per_epoch == 0 or epoch == 1 or epoch == 2 or epoch == 3:
                    save_plot(y_enc.squeeze().cpu(),
                              f'{log_dir}/generated_enc_{i}_epoch{epoch}.png')
                    save_plot(y_dec.squeeze().cpu(),
                              f'{log_dir}/generated_dec_{i}_epoch{epoch}.png')
                    save_plot(attn.squeeze().cpu(),
                              f'{log_dir}/alignment_{i}_epoch{epoch}.png')
                    # synthesize audio
                    audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                    ## create folder if not exist
                    write(f'{log_dir}/sample_{i}_epoch{epoch}.wav', 22050, audio)


        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []

        with tqdm(loader, total=len(train_dataset) // batch_size) as progress_bar:
            for batch in progress_bar:
                model.zero_grad()
                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                spk = batch['spk'].cuda()
                #emo = batch['emo'].cuda()

                melstyle = batch["melstyle"].cuda()
                melstyle_len = batch["melstyle_lengths"].cuda()  # Use it rather than x_mask
                emo_label = batch["emo_label"].cuda()

                dur_loss, prior_loss, diff_loss = model.compute_loss(x,
                                                                     x_lengths,
                                                                     y,
                                                                     y_lengths,
                                                                     spk=spk,
                                                                     out_size=out_size,
                                                                     emo=emo,
                                                                     #psd=(pit, eng, dur),
                                                                     melstyle=melstyle,
                                                                     emo_label=emo_label
                                                                     )
                loss = sum([dur_loss, prior_loss, diff_loss])
                loss.backward()

                # clip the gradience
                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()

                logger.add_scalar('training/duration_loss', dur_loss,
                                  global_step=iteration)
                logger.add_scalar('training/prior_loss', prior_loss,
                                  global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss,
                                  global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)

                msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                progress_bar.set_description(msg)
                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                iteration += 1

        msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
        msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(msg)

        if epoch % save_every > 0:
            continue

        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")


from typing import Sequence
from typing import Union
from pathlib import Path


def resume(
    checkpoint: Union[str, Path],
    model: torch.nn.Module,
    ngpu: int = 0,
):
    states = torch.load(
        checkpoint,
        map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu",
    )
    model.load_state_dict(states)
    """
        for optimizer, state in zip(optimizers, states["optimizers"]):
        optimizer.load_state_dict(state)
    """


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


if __name__ == "__main__":
    import argparse
    config_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/config/ESD"

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=250000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        #required=True,
        help="path to preprocess.yaml",
        default=config_dir + "/preprocess_gradTTS.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str,
        #required=True,
        help="path to model.yaml",
        default=config_dir + "/model_gradTTS_v2.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str,
        #required=True,
        help="path to train.yaml",
        default=config_dir + "/train_gradTTS.yaml"
    )
    args = parser.parse_args()

    # Train on ljspeech dataset
    if False:
        train_filelist_path = 'resources/filelists/ljspeech/train.txt'
        valid_filelist_path = 'resources/filelists/ljspeech/valid.txt'
        log_dir = "logs/condGradTTS"
        train_process(train_filelist_path, valid_filelist_path, log_dir=log_dir)

    # Train on ESD dataset
    if True:
        # Input: Config
        preprocess_config = yaml.load(
            open(args.preprocess_config, "r"), Loader=yaml.FullLoader
        )
        model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
        train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
        configs = (preprocess_config, model_config, train_config)
        # Output
        # log_dir: "./logs/crossatt_diffuser/"
        train_process_cond(configs)