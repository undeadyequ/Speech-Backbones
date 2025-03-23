1/28
- not viewable
- torch selection
- pycharm usage


gate_msa mean: tensor(0.5911, device='cuda:0')
gate_mocha mean: tensor(0.1017, device='cuda:0')
gate_mlp mean: tensor(0.8358, device='cuda:0')
gate_msa mean: tensor(0.5994, device='cuda:0')
gate_mocha mean: tensor(0.1524, device='cuda:0')
gate_mlp mean: tensor(1.6402, device='cuda:0')



K R IY1 M Z W IH0 TH sil P IH1 NG K EH1 JH IH0 Z

116, 130, 113, 118, 146, 144, 108, 134, 129, 109, 120, 116, 94, 115, 108, 146



- Data define in evalobj_prosody_simiarity
- 
"""python
cross_attn: [[h, b, t_t, t_s] * layers * t_sel]  ->  [t_sel, layers. h, b, t_t, t_s]
attn_map_dict: {"model1": {"emo1": cross_attn}}}

cross_attn_vis: [t_t, t_s]   # t_sel, layers. h is given
attn_map_for_vis: {"model1": {"emo1": cross_attn_vis}}}
"""

"""python
1100000000
0011100000
0000011111
"""

"""python

muy_dur = ""
ref_dur = ""

muy_dur_seq = dur2seq(muy_dur)
ref_dur_seq = dur2seq(ref_dur)
poneme_diag = torch.mul(muy_dur_seq, ref_dur_seq)


"""::


monAttn_loss: 0.0:  14%|█▍        | 34/241 [00:10<01:04,  3.22it/s]
1. i, dur, searchpFrameLen, indxs 0 tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward0>) tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
         14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
         28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
         42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
         56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
         70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
         84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
         98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
        126, 127], device='cuda:0') tensor([118, 119, 120, 121, 122, 123, 124], device='cuda:0')