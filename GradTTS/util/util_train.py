from typing import Any, Dict, List, Optional, Tuple, Union
import torch

import yaml
from GradTTS.text.symbols import symbols
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm
def log_tts_output(enc_out, dec_out, attns, mode="train"):
    pass

def log_loss(losses, epoch, iteration, logger):
    """
    logger to losser and append them to msag
    Args:
        dur_loss:
        prior_loss:
        diff_loss:
        monAttn_loss:
        commit_loss:
        vq_loss:

    Returns:

    """
    dur_loss, prior_loss, diff_loss, monAttn_loss, commit_loss, vq_loss = losses

    logger.add_scalar('training/duration_loss', dur_loss, global_step=iteration)
    logger.add_scalar('training/prior_loss', prior_loss, global_step=iteration)
    logger.add_scalar('training/diffusion_loss', diff_loss, global_step=iteration)
    logger.add_scalar('training/monAttn_loss', monAttn_loss, global_step=iteration)
    logger.add_scalar('training/vq_loss', vq_loss, global_step=iteration)

    msg = (f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, '
           f'prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}, ')
    if torch.is_nonzero(monAttn_loss.item()):
        msg += f'monAttn_loss: {monAttn_loss.item()}'
    if torch.is_nonzero(vq_loss.item()):
        msg += f'commit_loss: {commit_loss.item()}, vq_loss: {vq_loss.item()}'
    return msg