import sys
sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS')
sys.path.append('/home/rosen/Project/Speech-Backbones/GradTTS/hifi-gan/')
from GradTTS.model.maskCreation import create_p2p_mask, DiffAttnMask

def test_create_p2p_mask():
    p2p_mask = create_p2p_mask(
        2,
        5,
        dims=4,
        tgt_range=(0, 1),
        ref_range=(2, 4)
    )

def test_diffAttnMask():
    """

    Test create_p2p_mask2
        down  size: [torch.Size([1, 1, 160, 6]), torch.Size([1, 1, 80, 3]), torch.Size([1, 1, 40, 1]))]
        mid  size: [torch.Size([1, 1, 40, 1])]
        up  size: [torch.Size([1, 1, 80, 2]), torch.Size([1, 1, 160, 4])]
    """
    dffatnnMask = DiffAttnMask()
    dffmask_dict = dffatnnMask.create_p2p_mask(
        2,
        6,  # test odds (6) or even (8)
        mask_dims=4,
        tgt_range=(0, 1),
        ref_range=(2, 4)
    )
    print(dffmask_dict)
    for item, l in dffmask_dict.items():
        print(item, " size:", [l_i.shape for l_i in l])