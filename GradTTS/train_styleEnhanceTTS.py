# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
import sys
sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS/model')
#sys.path.append('/Users/luoxuan/Project/tts/Speech-Backbones')
#sys.path.append('/Users/luoxuan/Project/tts/Speech-Backbones/GradTTS')

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#import params
from model import CondGradTTSDIT
from data import TextMelSpeakerEmoDataset, TextMelSpeakerEmoBatchCollate
from utils import plot_tensor, save_plot
from text.symbols import symbols
import yaml
from model.utils import fix_len_compatibility
sys.path.append('./hifi-gan/')
from scipy.io.wavfile import write
from GradTTS.read_model import get_vocoder
from typing import Union
from pathlib import Path

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
    out_size = fix_len_compatibility(2 * sample_rate // 256)  # 128
    preprocess_dir = preprocess_config["path"]["preprocessed_path"]
    train_filelist_path = preprocess_dir + "/" + preprocess_config["path"]["index_train_f"]
    valid_filelist_path = preprocess_dir + "/" + preprocess_config["path"]["index_val_f"]
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
    psd_gran = preprocess_config["psd_gran"]
    need_rm_sil = preprocess_config["need_rm_sil"]

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
    melstyle_n = int(model_config["decoder"]["melstyle_n"])
    psd_n = int(model_config["decoder"]["psd_n"])


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
    show_img_per_epoch = float(train_config["show_img_per_epoch"])

    # dit_mocha config
    dit_mocha = model_config["dit_mocha"]
    stdit = model_config["stdit"]
    stditMocha = model_config["stditMocha"]

    guided_attn = model_config["loss"]["guided_attn"]
    diff_model = model_config["diff_model"]
    ref_encoder = model_config["ref_encoder"]
    ref_embedder = model_config["ref_embedder"]

    # vqvae config
    tvencoder = model_config["tv_encoder"]


    # dataset
    train_dataset = TextMelSpeakerEmoDataset(train_filelist_path,
                                             meta_json_path,
                                             cmudict_path,
                                             preprocess_dir,
                                             add_blank,
                                             n_fft, n_feats,
                                             sample_rate, hop_length,
                                             win_length, f_min, f_max,
                                             datatype=ref_embedder,
                                             need_rm_sil=need_rm_sil
                                             )
    test_dataset = TextMelSpeakerEmoDataset(valid_filelist_path,
                                            meta_json_path,
                                            cmudict_path,
                                            preprocess_dir,
                                            add_blank,
                                            n_fft, n_feats,
                                            sample_rate, hop_length,
                                            win_length, f_min, f_max,
                                            datatype=ref_embedder,
                                            need_rm_sil=need_rm_sil
                                            )

    batch_collate = TextMelSpeakerEmoBatchCollate()

    loader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        collate_fn=batch_collate,
                        drop_last=True,
                        num_workers=8,
                        shuffle=True)

    # get test_batch size
    test_batch = test_dataset.sample_test_batch(size=4)

    # test condition
    model = CondGradTTSDIT(nsymbols,   # CHANGE to config
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
                        tvencoder_config=tvencoder
                        ).cuda()
    with open(f'{log_dir}/model.log', 'a') as f:
        f.write(str(model))
        f.write('Number of encoder parameters = %.2fm' % (model.encoder.nparams / 1e6))
        f.write('Number of decoder parameters = %.2fm' % (model.decoder.nparams / 1e6))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    if resume_epoch > 1:
        resume(ckpt, model, optimizer, 1)

    # Create subdir watchImg/model/samples
    Path("{}/img".format(log_dir)).mkdir(exist_ok=True)
    Path("{}/models".format(log_dir)).mkdir(exist_ok=True)
    Path("{}/samples".format(log_dir)).mkdir(exist_ok=True) # ??

    print('Logging test batch...')
    for item in test_batch:
        mel, spk = item['y'], item['spk']
        i = int(spk.cpu())
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{log_dir}/img/original_{i}.png')

    print('Start training...')
    iteration = 0
    vocoder = get_vocoder()

    for epoch in range(resume_epoch + 1, n_epochs + 1):
        model.eval()
        print('Synthesis...')
        with (torch.no_grad()):
            for j, item in enumerate(test_batch):
                break
                x = item['x'].to(torch.long).unsqueeze(0).cuda()
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                spk = item['spk'].to(torch.long).cuda()
                emo_label = item["emo_label"].cuda()
                melstyle = item["melstyle"].cuda()  # (b, d, l)
                melstyle_lengths = torch.LongTensor([melstyle.shape[-1]]).cuda()
                i = "spk" + str(spk.item()) + "_emo" + str(emo_label.item()) + "_txt" + str(j)

                model_input = {
                    "x": x,
                    "x_lengths": x_lengths,
                    "n_timesteps": n_timesteps,
                    "temperature": temperature, #
                    "stoc": stoc,
                    "spk": spk,
                    "length_scale": length_scale,
                    # "emo": torch.randn(batch, style_emb_dim),
                    # "melstyle": torch.randn(batch, style_emb_dim, mel_max_len),# non-vae
                    "melstyle": melstyle,
                    "melstyle_lengths": melstyle_lengths,
                    "emo_label": emo_label
                }
                y_enc, y_dec, mas_attn, self_attns_list, cross_attns_list = model(**model_input)

                # evaluate result
                ## encoder/dec ouput
                logger.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                ## CrossAttention map
                #logger.add_image(f'image_{i}/alignment',
                #                 plot_tensor(cros_attn_00.squeeze().cpu()),
                #                 global_step=iteration, dataformats='HWC')
                ## show image and audio (show initiate image to gheck earlierly)
                if epoch % show_img_per_epoch == 0 or epoch == 1 or epoch == 2 or epoch == 3:
                    save_plot(y_enc.squeeze().cpu(),
                              f'{log_dir}/img/generated_enc_{i}_epoch{epoch}.png')
                    save_plot(y_dec.squeeze().cpu(),
                              f'{log_dir}/img/generated_dec_{i}_epoch{epoch}.png')
                    if len(cross_attns_list) != 0:
                        t1, b1, bt1, h1 = 0, 1, 0, 0  # time, block_n, batch, head
                        t2, b2, bt2, h2 = 5, 5, 0, 0  # (0, 10, 20, 30, 40, 49), layer = 6
                        cros_attn_00 = cross_attns_list[t1, b1, bt1, h1]
                        cros_attn_55 = cross_attns_list[t2, b2, bt2, h2]
                        save_plot(cros_attn_00.squeeze().cpu(),
                                  f'{log_dir}/img/cross_attn_t0_b0_{i}_epoch{epoch}.png')
                        save_plot(cros_attn_55.squeeze().cpu(),
                                  f'{log_dir}/img/cross_attn_t49_b6_{i}_epoch{epoch}.png')

                    #save_plot(show_attn.squeeze().cpu(),
                    #          f'{log_dir}/img/alignment_bern_{i}_epoch{epoch}.png')
                    # synthesize audio
                    audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                    ## create folder if not exist
                    write(f'{log_dir}/samples/sample_{i}_epoch{epoch}.wav', 22050, audio)
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        monAttn_losses = []
        commit_losses = []
        vq_losses = []

        with tqdm(loader, total=len(train_dataset) // batch_size) as progress_bar:
            for batch in progress_bar:
                model.zero_grad()
                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                spk = batch['spk'].cuda()
                #emo = batch['emo'].cuda()
                melstyle = batch["melstyle"].cuda()
                melstyle_lengths = batch["melstyle_lengths"].cuda()  # Use it rather than x_mask
                emo_label = batch["emo_label"].cuda()

                #if datatype == "psd" and psd_gran == "frame":
                #    dur = item["dur"].to(torch.long).squeeze(0).cuda()
                #    melstyle = torch.repeat_interleave(melstyle, repeats=dur, dim=2)
                #    assert melstyle.shape[2] == y.shape[2]
                #if x.shape[1] != melstyle[].shape[2]:
                #    print("x {} and melstyle {} should have same len".format(x.shape[1], melstyle.shape[2]))

                model_input = {
                    "x": x,  # (b, p_l)
                    "x_lengths": x_lengths, # (b,)
                    "y": y,  # (b, mel_dim, mel_l)
                    "y_lengths": y_lengths, # (b)
                    "spk": spk,  # (b)
                    "out_size": out_size,
                    "melstyle": melstyle,  # (b, ls_dim, ls_l)
                    "melstyle_lengths": melstyle_lengths,
                    "emo_label": emo_label  # (b)
                }

                #[print(k, v.shape) for k, v in model_input.rm_items() if torch.is_tensor(v)]
                dur_loss, prior_loss, diff_loss, monAttn_loss, commit_loss, vq_loss = model.compute_loss(**model_input)

                #loss = sum([dur_loss, prior_loss, diff_loss, vq_loss])
                loss = sum([dur_loss, prior_loss, diff_loss, monAttn_loss, vq_loss])

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
                logger.add_scalar('training/monAttn_loss', monAttn_loss,
                                  global_step=iteration)
                logger.add_scalar('training/vq_loss', vq_loss,
                                  global_step=iteration)

                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)
                #msg = (f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, '
                #       f'prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}, '
                #       f'monAttn_loss: {monAttn_loss.item()}, commit_loss: {commit_loss.item()}, '
                #       f'vq_loss: {vq_loss.item()}')

                msg = (f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, '
                       f'prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}, '
                       f'monAttn_loss: {monAttn_loss.item()}, '
                       #f'commit_loss: {commit_loss.item()}, '
                       f'vq_loss: {vq_loss.item()}')

                progress_bar.set_description(msg)
                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                monAttn_losses.append(monAttn_loss.item())
                commit_losses.append(commit_loss.item())
                vq_losses.append(vq_loss.item())
                iteration += 1

        msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
        msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        msg += '| monAttn loss = %.3f\n' % np.mean(monAttn_losses)
        msg += '| commit loss = %.3f\n' % np.mean(commit_losses)
        msg += '| vq loss = %.3f\n' % np.mean(vq_losses)

        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(msg)
        if epoch % save_every > 0:
            continue

        # save ckpt
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": {
                    "dur_loss": dur_loss,
                    "prior_loss": prior_loss,
                    "diff_loss": diff_loss
                },
            },
            f"{log_dir}/models/grad_{epoch}.pt"
        )
        #ckpt = model.state_dict()
        #torch.save(ckpt, f=f"{log_dir}/models/grad_{epoch}.pt")

def resume(
    checkpoint: Union[str, Path],
    model: torch.nn.Module,
    optimzier: None,
    ngpu: int = 0,
):
    ckpt_states = torch.load(
        checkpoint,
        map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu",
    )
    if "model_state_dict" in ckpt_states:
        model.load_state_dict(
            ckpt_states["model_state_dict"])
        optimzier.load_state_dict(
            ckpt_states["optimizer_state_dict"]
        )
    else:
        states = torch.load(
            checkpoint,
            map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu",
        )
        model.load_state_dict(states)


if __name__ == "__main__":
    import argparse
    config_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/config/ESD"
    #config_dir = "/Users/luoxuan/Project/tts/Speech-Backbones/GradTTS/config/ESD"
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=250000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        #required=True,
        help="path to preprocess.yaml",
        default=config_dir + "/preprocess_styleAlignedTTS.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str,
        #required=True,
        help="path to model.yaml",
        default=config_dir + "/model_styleAlignedTTS_guidloss_codec.yaml",
        #default = config_dir + "/model_gradTTS_linear.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str,
        #required=True,
        help="path to train.yaml",
        default=config_dir + "/train_styleAlignedTTS.yaml"
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