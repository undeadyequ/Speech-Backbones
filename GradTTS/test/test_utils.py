from GradTTS.model.utils import sequence_dur
import torch

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



if __name__ == '__main__':
    test_sequence_dur()