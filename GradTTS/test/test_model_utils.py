from GradTTS.model.utils import *
import torch

def test_generate_diagal_fromMask():
    attn_maps = torch.tensor([[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]])
    diags_attn_mpas = generate_diagal_fromMask(attn_maps)
    print("previous attn_maps", attn_maps)
    print("after attn_maps", diags_attn_mpas)


def test_all():
    dur1 = torch.Tensor([2, 4])
    dur2 = torch.Tensor([3, 2])

    #dur12_mask = generate_gd_diagMask(dur1, dur2)
    #print(dur12_mask)

    dur1 = torch.Tensor(
        [[0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0],
         [1, 1, 1, 2, 2, 3, 3, 3, 4, 0, 0]]  # after cut, may not from first phoneme
    )

    diag_dur_gd = torch.Tensor([
        [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    ]])

    diag_dur = get_diag_from_dur2(dur1)  #
    print(diag_dur)
    #assert torch.equal(diag_dur, diag_dur_gd)

if __name__ == '__main__':
    test_generate_diagal_fromMask()