import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import sys
import numpy as np
from tqdm import tqdm
import yaml
from scipy.io.wavfile import write
from typing import Union
from pathlib import Path

sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS/model')
#sys.path.append('/Users/luoxuan/Project/tts/Speech-Backbones')
#sys.path.append('/Users/luoxuan/Project/tts/Speech-Backbones/GradTTS')
sys.path.append('./hifi-gan/')

from GradTTS.util.util_config import convert_item2input
from GradTTS.util.util_train import log_tts_output
from GradTTS.read_model import get_vocoder
from GradTTS.model import CondGradTTSDIT3
from GradTTS.model.utils import fix_len_compatibility
from data import TextMelSpeakerEmoDataset, TextMelSpeakerEmoBatchCollate
from utils import plot_tensor, save_plot
from text.symbols import symbols
import time

emoN_emo = {
    "0": "ang",
    "1": "sur",
    "2": "sad",
    "3": "neu",
    "4": "hap"
}

txtN_para = {
    "0": "unpara",
    "1": "para",
    "2": "unpara",
    "3": "para",
    "4": "unpara",
    "5": "para"
}
torch.set_printoptions(threshold=100000)

def read_dataloder(
        preprocess_config,
        train_config,
        model_config,
        test_batch_size=4
    ):
    batch_size = int(train_config["batch_size"])
    # dynamic parameter
    add_blank = preprocess_config["feature"]["add_blank"]
    sample_rate = preprocess_config["feature"]["sample_rate"]
    # dataset path
    preprocess_dir = preprocess_config["path"]["preprocessed_path"]
    train_filelist_path = preprocess_dir + "/" + preprocess_config["path"]["index_train_f"]
    valid_filelist_path = preprocess_dir + "/" + preprocess_config["path"]["index_val_f"]
    meta_json_path = preprocess_dir + "/metadata_new.json"

    # dataset related
    cmudict_path = preprocess_config["path"]["cmudict_path"]
    n_fft = int(preprocess_config["feature"]["n_fft"])
    n_feats = int(preprocess_config["feature"]["n_feats"])
    hop_length = int(preprocess_config["feature"]["hop_length"])
    win_length = int(preprocess_config["feature"]["win_length"])
    f_min = int(preprocess_config["feature"]["f_min"])
    f_max = int(preprocess_config["feature"]["f_max"])
    ref_type = model_config["datatype"]
    need_rm_sil = preprocess_config["need_rm_sil"]

    # dataset
    train_dataset = TextMelSpeakerEmoDataset(train_filelist_path,
                                             meta_json_path,
                                             cmudict_path,
                                             preprocess_dir,
                                             add_blank,
                                             n_fft, n_feats,
                                             sample_rate, hop_length,
                                             win_length, f_min, f_max,
                                             datatype=ref_type,
                                             need_rm_sil=need_rm_sil)
    test_dataset = TextMelSpeakerEmoDataset(valid_filelist_path,
                                            meta_json_path,
                                            cmudict_path,
                                            preprocess_dir,
                                            add_blank,
                                            n_fft, n_feats,
                                            sample_rate, hop_length,
                                            win_length, f_min, f_max,
                                            datatype=ref_type,
                                            need_rm_sil=need_rm_sil
                                            )
    batch_collate = TextMelSpeakerEmoBatchCollate()

    train_loader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        collate_fn=batch_collate,
                        drop_last=True,
                        num_workers=8,
                        shuffle=True)
    test_batch = test_dataset.sample_test_batch(size=test_batch_size)

    return train_dataset, train_loader, test_dataset, test_batch


def train_process_cond(configs, args):
    preprocess_config, model_config, train_config = configs
    torch.manual_seed(train_config["seed"])

    log_dir = train_config["path"]["log_dir"]
    np.random.seed(train_config["seed"])
    logger = SummaryWriter(log_dir=log_dir)

    # train parameter
    batch_size = int(train_config["batch_size"])
    learning_rate = float(train_config["learning_rate"])
    n_epochs = int(train_config["n_epochs"])
    resume_epoch = int(train_config["resume_epoch"])
    save_every = int(train_config["save_every"])
    ckpt = f"{log_dir}models/grad_{resume_epoch}.pt"
    show_img_per_epoch = float(train_config["show_img_per_epoch"])
    add_blank = preprocess_config["feature"]["add_blank"]
    nsymbols = len(symbols) + 1 if add_blank else len(symbols)
    sample_rate = preprocess_config["feature"]["sample_rate"]
    out_size = fix_len_compatibility(2 * sample_rate // 256)  # 128

    train_dataset, train_loader, test_dataset, test_batch = read_dataloder(preprocess_config, train_config,
        model_config, test_batch_size=4)

    # Initialize model
    fine_adjust_config(model_config, args)
    condGradTTSDIT_configs = dict(n_vocab=nsymbols, **model_config["stditCross"])
    model = CondGradTTSDIT3(**condGradTTSDIT_configs).cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    vocoder = get_vocoder()
    if resume_epoch > 1:
        resume(ckpt, model, optimizer, 1)

    # log file
    with open(f'{log_dir}/model.log', 'a') as f:
        f.write(time.ctime())  # 'Mon Oct 18 13:35:29 2010'
        f.write(str(model))
        f.write('Number of encoder parameters = %.2fm' % (model.encoder.nparams / 1e6))
        #f.write('Number of ref_encoder parameters = %.2fm' % (model.ref_encoder.nparams / 1e6))
        f.write('Number of decoder parameters = %.2fm' % (model.decoder.nparams / 1e6))
        f.write("###############configs##################")
        f.write(json.dumps(condGradTTSDIT_configs))
    # Create subdir watchImg/model/samples
    Path("{}/img".format(log_dir)).mkdir(exist_ok=True)
    Path("{}/models".format(log_dir)).mkdir(exist_ok=True)
    Path("{}/samples".format(log_dir)).mkdir(exist_ok=True)

    ## logger
    print('Logging test batch...')
    for item in test_batch:
        mel, spk = item['y'], item['spk']
        i = int(spk.cpu())
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()), global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{log_dir}/img/original_{i}.png')

    print('Start training...')

    iteration = 0
    for epoch in range(resume_epoch + 1, n_epochs + 1):
        model.eval()
        print('Synthesis...')
        with ((torch.no_grad())):
            for j, item in enumerate(test_batch):
                model_input = convert_item2input(
                    item, model_config["inference"], mode="test")
                y_enc, y_dec, mas_attn, self_attns_list, cross_attns_list, _, _ = model(**model_input)

                i = "spk" + str(model_input["spk"].item()) + "_" + emoN_emo[str(model_input["emo_label"].item())] + \
                    "_" + txtN_para[str(j)]
                ## encoder/dec ouput
                logger.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
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
        dur_losses, prior_losses, diff_losses, monAttn_losses, commit_losses, vq_losses = [], [], [], [], [], []

        with tqdm(train_loader, total=len(train_dataset) // batch_size) as progress_bar:
            for batch in progress_bar:
                model.zero_grad()
                model_input = convert_item2input(
                    batch, train_param=dict(out_size=out_size), mode="train")

                #[print(k, v.shape) for k, v in model_input.rm_items() if torch.is_tensor(v)]
                dur_loss, prior_loss, diff_loss, monAttn_loss, commit_loss, vq_loss = model.compute_loss(**model_input)

                loss = sum([dur_loss, prior_loss, diff_loss, monAttn_loss, vq_loss])
                loss.backward()

                # clip the gradience
                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1)
                optimizer.step()

                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm, global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm, global_step=iteration)
                #log_loss((dur_loss, prior_loss, diff_loss, monAttn_loss, commit_loss, vq_loss), epoch, iteration, logger)

                logger.add_scalar('training/duration_loss', dur_loss, global_step=iteration)
                logger.add_scalar('training/prior_loss', prior_loss, global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss, global_step=iteration)
                logger.add_scalar('training/monAttn_loss', monAttn_loss, global_step=iteration)
                logger.add_scalar('training/vq_loss', vq_loss, global_step=iteration)


                msg = (f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, '
                       f'prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}, ')

                if condGradTTSDIT_configs["guide_loss"]:
                    msg += f'monAttn_loss: {monAttn_loss.item()}'
                if "vae" in condGradTTSDIT_configs["ref_encoder_type"]:
                    msg += f'commit_loss: {commit_loss.item()}, vq_loss: {vq_loss.item()}'

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

        if condGradTTSDIT_configs["guide_loss"]:
            msg += '| monAttn loss = %.3f\n' % np.mean(monAttn_losses)
        if "vae" in condGradTTSDIT_configs["ref_encoder_type"]:
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
            f"{log_dir}/models/grad_{epoch}.pt")

def resume(
    checkpoint: Union[str, Path],
    model: torch.nn.Module,
    optimzier: None,
    ngpu: int = 0,
):
    ckpt_states = torch.load(
        checkpoint,
        map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu",)
    if "model_state_dict" in ckpt_states:
        model.load_state_dict(
            ckpt_states["model_state_dict"])
        optimzier.load_state_dict(
            ckpt_states["optimizer_state_dict"])
    else:
        states = torch.load(
            checkpoint,
            map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu",)
        model.load_state_dict(states)

def fine_adjust_config(model_config, args):
    model_config["stditCross"]["qk_norm"] = args.qk_norm
    model_config["stditCross"]["mono_hard_mask"] = args.mono_hard_mask
    model_config["stditCross"]["mono_mas_mask"] = args.mono_mas_mask
    model_config["stditCross"]["guide_loss"] = args.guide_loss
    model_config["stditCross"]["decoder_config"]["stdit_config"]["phoneme_RoPE"] = args.phoneme_RoPE
    model_config["stditCross"]["global_norm"] = args.global_norm
    model_config["stditCross"]["ref_encoder_type"] = args.ref_encoder_type

    if args.ref_encoder_type == "vae":
        model_config["stditCross"]["text_encoder_config"] = model_config["stditCross"]["ref_encoder_vae"]
    elif args.ref_encoder_type == "vaeEma":
        model_config["stditCross"]["text_encoder_config"] = model_config["stditCross"]["ref_encoder_vaeEma"]
    elif args.ref_encoder_type == "vaeGRL":
        model_config["stditCross"]["text_encoder_config"] = model_config["stditCross"]["ref_encoder_vaeGRL"]
    else:
        print("not support ref_encoder {}".format(args.ref_encoder_type))

if __name__ == "__main__":
    import argparse
    config_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/config/ESD"
    #config_dir = "/Users/luoxuan/Project/tts/Speech-Backbones/GradTTS/config/ESD"
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=250000)
    parser.add_argument(
        "-p", "--preprocess_config", type=str,
        #required=True,
        help="path to preprocess.yaml", default=config_dir + "/preprocess_stditCross.yaml")
    parser.add_argument(
        "-m", "--model_config", type=str,
        #required=True,
        # default = config_dir + "/model_gradTTS_linear.yaml",
        help="path to model.yaml", default=config_dir + "/model_stditCross.yaml")
    parser.add_argument(
        "-t", "--train_config", type=str,
        # required=True,
        help="path to train.yaml", default=config_dir + "/train_stditCross.yaml")

    # fine_adjust option
    parser.add_argument("-qk", "--qk_norm", type=bool,default=False)
    parser.add_argument("-hMask", "--mono_hard_mask", type=bool, default=False)
    parser.add_argument("-mMask", "--mono_mas_mask", type=bool, default=False)
    parser.add_argument("-gLoss", "--guide_loss", type=bool, default=True)
    parser.add_argument("-pRoPE", "--phoneme_RoPE", type=bool, default=True)
    parser.add_argument("-gNorm", "--global_norm", type=bool, default=False)
    parser.add_argument("-reType", "--ref_encoder_type", type=str, default="mlp")

    args = parser.parse_args()

    # Train on ESD dataset
    # Input: Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    # Output
    # log_dir: "./logs/crossatt_diffuser/"
    train_process_cond(configs, args)