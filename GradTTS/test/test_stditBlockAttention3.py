import torch

from GradTTS.model.estimator.stditBlockAttention3 import enhancePhoneAttention


def test_enhancePhoneAttention():

    attn = torch.softmax(torch.randn((2, 1, 6, 6)), dim=-1)
    print(attn)
    refenh_ind_dur, synenh_ind_dur = (2, 4), (2, 4)
    enh_score = 1.0
    enh_type = "soft"

    enhancePhoneAttention(attn, refenh_ind_dur, synenh_ind_dur, enh_score, enh_type)