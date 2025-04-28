from GradTTS.model.utils import get_diag_from_dur2, generate_diagal_fromMask, create_p2pmask_from_seqdur
import torch

def test_generate_diagal_fromMask():
    attn_maps = torch.tensor([[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]])
    diags_attn_mpas = generate_diagal_fromMask(attn_maps)
    print("previous attn_maps", attn_maps)
    print("after attn_maps", diags_attn_mpas)


def test_get_diag_from_dur2l():
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


def test_create_p2pmask_from_seqdur():
    dur1 = torch.Tensor(
        [[0, 0, 0, 1, 1, 2, 2],
         [1, 1, 1, 2, 2, 0, 0]]  # after cut, may not from first phoneme
    )
    dur2 = torch.Tensor(
        [[0, 0, 1, 1, 2, 2],
         [1, 1, 2, 2, 0, 0]]  # after cut, may not from first phoneme
    )

    p2p_mask = create_p2pmask_from_seqdur(dur1, dur2)  #
    p2p_mask_gd =[
        [[1, 1, 1, 0, 0, 0, 0],
         [1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 1, 1]],

        [[1, 1, 1, 0, 0, 0, 0],
         [1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]],
        ]
    print(p2p_mask)

if __name__ == '__main__':
    test_generate_diagal_fromMask()