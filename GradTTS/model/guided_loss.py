import time

import numpy as np
import torch
import torch.nn.functional as F
from GradTTS.model.utils import make_non_pad_mask

class GuidedAttentionLoss(torch.nn.Module):
    """Guided attention loss function module.

    This module calculates the guided attention loss described in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_, which forces the attention to be diagonal.

    .. _`Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969

    """

    def __init__(self, sigma=0.4, alpha=1.0, reset_always=True):
        """Initialize guided attention loss module.

        Args:
            sigma (float, optional): Standard deviation to control how close attention to a diagonal.
            alpha (float, optional): Scaling coefficient (lambda).
            reset_always (bool, optional): Whether to always reset masks.

        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def create_guide_mask(self, ilens, olens, fix_len=None, chunksize=5):
        """

        Args:
            ilens: (b, )
            olens: (b, )
            fix_len:
            chunksize: int
        Returns:
            guided_attn_masks (b, ilens_max, olens_max)
            masks: (b, ilens_max, olens_max)
        """

        guided_attn_masks = self._make_guided_attention_masks(ilens, olens, fix_len, chunksize).cuda()
        masks = self._make_masks(ilens, olens).cuda()
        return guided_attn_masks, masks

    def compute_mLoss(self, att_ws, guided_attn_masks, masks):
        losses = guided_attn_masks * att_ws
        masks_pad = torch.zeros(losses.shape, dtype=torch.bool).cuda()
        b, ilen, olen = masks.shape
        masks_pad[:, :, :, :ilen, :olen] = masks
        loss = torch.mean(losses.masked_select(masks_pad))
        return loss


    def forward(self, att_ws, ilens, olens, guided_attn_masks=None, fix_len=None, chunksize=6):
        """Calculate forward propagation.

        Args:
            att_ws (Tensor): Batch of attention weights (B, [h] ,T_max_out, T_max_in) or
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).

        Returns:
            Tensor: Guided attention loss value.
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens, fix_len, chunksize).to(att_ws.device)

            #if len(att_ws.shape) == 4:
            #    self.guided_attn_masks = self.guided_attn_masks.unsqueeze(1)
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)
        #print(self.guided_attn_masks[0, :10, :10])
        losses = self.guided_attn_masks * att_ws
        masks_pad = torch.zeros(losses.shape, dtype=torch.bool).cuda()
        b, ilen, olen = self.masks.shape
        masks_pad[:, :ilen, :olen] = self.masks
        loss = torch.mean(losses.masked_select(masks_pad))
        temp_guide_mask = self.guided_attn_masks
        if self.reset_always:
            self._reset_masks()
        #return self.alpha * loss
        return self.alpha * loss, temp_guide_mask

    def _make_guided_attention_masks(self, ilens, olens, fix_size=None, chunksize=6):
        n_batches = len(ilens)
        if fix_size is None:
            max_ilen = int(max(ilens))
            max_olen = int(max(olens))
        else:
            max_ilen, max_olen = fix_size
        guided_attn_masks = torch.zeros((n_batches, max_ilen, max_olen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            ilen = int(ilen)
            olen = int(olen)
            #guided_mask = self._make_guided_attention_mask(ilen, olen, self.sigma)
            guided_attn_masks[idx, :ilen, :olen] = self._make_guided_attention_mask(ilen, olen, self.sigma, chunksize)
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma, chunksize=6):
        """Make guided attention mask.

        Examples:
            >>> guided_attn_mask =_make_guided_attention(5, 5, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([5, 5])
            >>> guided_attn_mask
            tensor([[0.0000, 0.1175, 0.3935, 0.6753, 0.8647],
                    [0.1175, 0.0000, 0.1175, 0.3935, 0.6753],
                    [0.3935, 0.1175, 0.0000, 0.1175, 0.3935],
                    [0.6753, 0.3935, 0.1175, 0.0000, 0.1175],
                    [0.8647, 0.6753, 0.3935, 0.1175, 0.0000]])
            >>> guided_attn_mask =_make_guided_attention(3, 6, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([6, 3])
            >>> guided_attn_mask
            tensor([[0.0000, 0.2934, 0.7506],
                    [0.0831, 0.0831, 0.5422],
                    [0.2934, 0.0000, 0.2934],
                    [0.5422, 0.0831, 0.0831],
                    [0.7506, 0.2934, 0.0000],
                    [0.8858, 0.5422, 0.0831]])

        """
        grid_x, grid_y = torch.meshgrid(torch.arange(ilen), torch.arange(olen))
        grid_x, grid_y = grid_x.float(), grid_y.float()
        guide_mask = 1.0 - torch.exp(-(grid_y / ilen - grid_x / olen) ** 2 / (2 * (sigma ** 2)))
        if chunksize > 0:
            for id, row in enumerate(guide_mask):
                min_value, min_index = torch.min(row).item(), torch.argmin(row).item()
                guide_mask[id, max(min_index - chunksize, 0) : min(min_index + chunksize, len(row))] = min_value
        return guide_mask

    @staticmethod
    def _make_masks(ilens, olens):
        """Make masks indicating non-padded part.

        Args:
            ilens (LongTensor or List): Batch of lengths (B,).
            olens (LongTensor or List): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor indicating non-padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens, olens = [5, 2], [8, 5]
            >>> _make_mask(ilens, olens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)

        """
        in_masks = make_non_pad_mask(ilens)  # (B, T_in)
        out_masks = make_non_pad_mask(olens)  # (B, T_out)
        return in_masks.unsqueeze(-1) & out_masks.unsqueeze(-2)  # (B, T_out, T_in)