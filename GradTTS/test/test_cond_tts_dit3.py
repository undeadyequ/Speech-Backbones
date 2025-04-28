from GradTTS.model.cond_tts_dit3 import synfdur2reffdur
import torch
from GradTTS.model.utils import pdur2fdur, scale_dur, pdur2syldur

def test_generate_input():
    """
    syn_pdur =
    ref_flen =
    syn_cut_flower =
    syn_cut_fupper =

    """
    pass


def test_synfdur2reffdur():
    syn_pdur = torch.LongTensor([[1, 3, 2],
                                [2, 4, 1]])
    y_lens = torch.Tensor(
        [[1, 1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 1, 1]])
    syn_fdur = pdur2fdur(syn_pdur, y_lens)

    target_seq_len = torch.Tensor([18, 13]) # test
    ref_pdur = scale_dur(syn_pdur, target_seq_len)

    ref_y_lens = torch.Tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
    ref_fdur = pdur2fdur(ref_pdur, ref_y_lens)

    print("syn_fdurs", syn_fdur)
    print("ref_fdurs", ref_fdur)

    syn_cut_flower, syn_cut_fupper = 3, 5
    #print("ref_pdur", ref_pdur)

    ref_fdur_cut = synfdur2reffdur(syn_cut_flower, syn_cut_fupper, syn_fdur, ref_fdur)

    print("below should have same phonemes")
    print("cut of syn_fdurs", syn_fdur[:, syn_cut_flower: syn_cut_fupper])
    print("ref_fdur_cut", ref_fdur_cut)