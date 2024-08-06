# if run under test folder
import sys

import numpy as np

sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS')

import torch
import yaml
from GradTTS.model.sampleGuidence import SampleGuidence, create_pitch_bin_mask
#import pytest


def test_others():
    # test mask
    attn = torch.ones([2, 3, 4, 5]) + 1
    masks = torch.zeros([2, 3, 4, 5])

    masks[:, :, :2, :] = 1
    attn_after = attn.masked_fill(masks>0, 0)

    print(attn)
    print(attn_after)