from GradTTS.model.utils import sequence_dur, scale_dur
import torch
from pymcd.mcd import Calculate_MCD


def test_sequence_dur():
    dur = [[1, 3, 2],
         [2, 4, 1]]
    dur = torch.LongTensor(dur)

    y_lens = torch.Tensor(
        [[1,1,1,1,1,1,0],
         [1,1,1,1,1,1,1]]
    )
    durs_seq = sequence_dur(dur, y_lens)


    durs_seq_gd = [
        [0, 1, 1, 1, 2, 2, 0],    # value = Index of phoneme,    times = nums of frame in this phoenme
        [0, 0, 1, 1, 1, 1, 2],
    ]

    print(durs_seq)
    #assert torch.LongTensor(durs_seq_gd) == durs_seq_gd


def test_scale_dur():
    ref_dur = [[1, 3, 2],
               [2, 4, 1]]
    ref_dur = torch.LongTensor(ref_dur)

    target_seq_len = torch.Tensor([18, 14]) # test1
    target_seq_len = torch.Tensor([18, 13]) # test2

    target_dur_seq = scale_dur(ref_dur, target_seq_len)

    """
    target_dur_seq_gd = [
        [3, 9, 6],    # value = Index of phoneme,    times = nums of frame in this phoenme
        [4, 8, 2],
    ]
    """

    target_dur_seq_gd = torch.Tensor(
        [[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0]]
    ).long()

    print(target_dur_seq)
    #assert torch.equal(target_dur_seq, target_dur_seq_gd)


def test_calculate_mcd():
    mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
    wav1 = "/home/rosen/Project/Speech-Backbones/GradTTS/exp/result50/stditCross_guide_codec/spk13_Angry_txt10.wav"
    wav2 = "/home/rosen/Project/Speech-Backbones/GradTTS/exp/result50/reference/spk13_Angry_txt10.wav"
    res = mcd_toolbox.calculate_mcd(wav1, wav2)
    print(res)



if __name__ == '__main__':
    #test_sequence_dur()
    test_scale_dur()