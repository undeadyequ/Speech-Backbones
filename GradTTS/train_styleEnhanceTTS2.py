import sys
#sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS/model')
sys.path.append('/Users/luoxuan/Project/tts/Speech-Backbones')
sys.path.append('/Users/luoxuan/Project/tts/Speech-Backbones/GradTTS')

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
from util.util_config import convert_originConfig_to_styleEnhanceConfig


def read_dataloder(
        preprocess_config,
        train_config,
        model_config
):
    """
    Process
    0. setup models
    1. get dataloader (dataset, batch_collate)
    2. get data sample
    """
    # 1. get dataloader (dataset, batch_collate)
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

    # train
    batch_size = int(train_config["batch_size"])
    learning_rate = float(train_config["learning_rate"])
    n_epochs = int(train_config["n_epochs"])
    resume_epoch = int(train_config["resume_epoch"])
    save_every = int(train_config["save_every"])
    ckpt = f"{log_dir}models/grad_{resume_epoch}.pt"

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

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              collate_fn=batch_collate,
                              drop_last=True,
                              num_workers=8,
                              shuffle=True)
    return train_loader, train_dataset, test_dataset


def train_styleEnhanceTTS(
        preprocess_config,
        train_config,
        model_config
):
    """
    Process
    3. feed to model, get loss, start step.
    4. save model and print ce_loss
    """
    # setup model
    model = CondGradTTSDIT(
        **convert_originConfig_to_emoMixConfig(
            train_config, preprocess_config, model_config))

    log_dir = train_config["path"]["log_dir"]
    learning_rate = float(train_config["learning_rate"])
    n_epochs = int(train_config["n_epochs"])
    resume_epoch = int(train_config["resume_epoch"])
    save_every = int(train_config["save_every"])
    ckpt = f"{log_dir}models/grad_{resume_epoch}.pt"
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    length_scale = float(model_config["encoder"]["length_scale"])
    n_timesteps = int(model_config["decoder"]["n_timesteps"])
    batch_size = int(train_config["batch_size"])

    train_loader, train_dataset, test_dataset = read_dataloder(
        preprocess_config,
        train_config,
        model_config
    )
    print("start training")
    iteration = 0

    model.train()

    # setup data_loader
    for epoch in range(resume_epoch + 1, n_epochs + 1):
        losses = []
        print(epoch)
        with tqdm(train_loader, total=len(train_dataset) // batch_size) as progress_bar:
            for batch in progress_bar:
                model.zero_grad()
                # 2. get data sample
                x, x_lengths, y, y_lengths, spk, emo_label = get_train_sample_from_batch(batch)
                # 3. feed to model, get loss, start step.
                dur_loss, prior_loss, diff_loss, style_loss = model.compute_loss(
                    x,
                    x_lengths,
                    y,
                    y_lengths,
                    length_scale,
                    spk,
                    emo_label,
                    n_timesteps,
                    out_size=100
                )

                loss = dur_loss + prior_loss + diff_loss + style_loss

                loss.backward()
                optimizer.step()
                losses.append(loss)

                msg = f"Epoch: {epoch}, iteration: {iteration} | ce_loss {loss}"
                print(msg)
                iteration += 1

        # 4. save model and log
        # msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
        msg += "Epoch %d: ce_loss = %.3f" % (epoch, np.mean(loss))
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": {
                    "ce_loss": np.mean(losses),
                    "dur_loss": np.mean(dur_loss),
                    "prior_loss": np.mean(prior_loss),
                    "diff_loss": np.mean(diff_loss),
                    "style_loss": np.mean(style_loss)
                },
            },
            f"{log_dir}/models/grad_{epoch}.gt"
        )


def get_train_sample_from_batch(batch, cuda=False):
    if cuda:
        x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
        y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
        spk = batch['spk'].cuda()
        # melstyle = batch["melstyle"].cuda()
        # melstyle_len = batch["melstyle_lengths"].cuda()
        emo_label = batch["emo_label"].cuda()
    else:
        x, x_lengths = batch['x'], batch['x_lengths']
        y, y_lengths = batch['y'], batch['y_lengths']
        spk = batch['spk']
        # melstyle = batch["melstyle"]
        # melstyle_len = batch["melstyle_lengths"]
        emo_label = batch["emo_label"]
    return x, x_lengths, y, y_lengths, spk, emo_label


if __name__ == '__main__':
    config_dir = "/Users/luoxuan/Desktop/diffTTS_benchmark/config/ESD"
    train_config = config_dir + "/train_gradTTS.yaml"
    preprocess_config = config_dir + "/preprocess_gradTTS.yaml"
    model_config = config_dir + "/model_gradTTS_v3.yaml"

    preprocess_config = yaml.load(
        open(preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)
    train_EMOmix(preprocess_config, train_config, model_config)