- [Cross Attention](#cross-attention)
- [p2p](#p2p)
  - [f2f:](#f2f)
  - [f2b](#f2b)
    - [The size of attn\_map](#the-size-of-attn_map)
  - [Mask shape](#mask-shape)
- [Problem](#problem)
- [Code](#code)
  - [Code defects](#code-defects)
  - [Wrong Code](#wrong-code)
  - [NEED-TO-Check](#need-to-check)
  - [Question](#question)
  - [expected](#expected)


# Cross Attention
Input: 
Q: (c, l_q, d_q)
K: (l_k, d_k)
V: (l_k, d_k)
(batch is ignored)

# p2p 

## f2f:

Q \
:arrow_right:Conv2d(c, h * att_dim): Q(h * att_dim, l_q, d_q) \
:arrow_right:View: Q(h, l_q, **d_q * att_dim**) \
(att_dim **varied by**: init_att_dim / layer_n, h and w also changed ?)

K \
:arrow_right:Linear(d_k, d_q * att_dim * h)\
:arrow_right:View: K(h, l_k, **d_q * att_dim**) 

V \
:arrow_right:Linear(d_k, d_q * att_dim * h)\
:arrow_right:View: V(h, l_k, **d_q * att_dim**)  <- d_q is variable (Note1)

**att**=Q * K: (h, l_q, l_k) \  <- **frame_att** \
QKV=att * (V:arrow_right:T): QKV(h, l_q, d_q * att_dim)

QKV \
:arrow_right:View QKV(l_q, d_q, h * att_dim) \
:arrow_right:linear(h * att_dim, c)\
:arrow_right:View: QKV(c, l_q, d_q)\
:arrow_right:Resnet:arrow_right:Mask:arrow_right:LayerNorm: QKV(c, l_q, d_q)


- Definiation
  - s (or head): 4
  - c: head_dim: 32
  - d_q: (80 ~ 240)
  - d_k: (frame_len, 80 * 3) including spk, melstyle, emo_label
  - melstyle: (frame_len, 768) -> (frame_len, 80)
  - spk: (80) -> (frame_len, 80)
  - emo_label: (80) -> (frame_len, 80)
  - Channels: [(3, 64), (64, 128), (128, 256)]
  - input_q in 
    - down: [(64, l_q, 80), (128, 40, 40), (256, 20, 20)]
    - mid: [(256, 20, 20)]
    - up: [(256, 20, 20), (128, 20, 20), (64, 40, 40)]

- Note
  1. **The last layer of downNet is not downsampled**.


## f2b
### The size of attn_map 
 - down: 
   - (b, c, l_q * d_q, l_k)       -> (b, 64, l_q * 80, l_k)
   - (b, c*2, l_q * d_q/2, l_k/2) -> (128, 40, 40)
   - (b, c*4, l_q * d_q/4, l_k/4)
 - mid: 
   - (b, c*4, l_q * d_q/4, l_k/4)
 - up: 
   - (b, c*4, l_q * d_q/4, l_k/4)
   - (b, c*2, l_q * d_q/4, l_k/4)
   - (b, c, l_q * d_q/2, l_k/2)


## Mask shape
x_mask: 
(1, 1, 1, 108)
(1, 1, 1, 54)
(1, 1, 1, 27)
(1, 1, 1, 14)

# Problem
- The phoneme-level styel not 
- 
- The attention map affect style
- The noise affect style
  - How to control
- Addding f2b attention map may get
  - Quality get bad
  - b2b style transfering reversly worse?
- The estimator is energy increasing function
- Increasing 

# Code
## Code defects
- Too many view/transpose in attention compuation

## Wrong Code 
- 1 of attn_mask means useless information

## NEED-TO-Check
1. the resnet and attn of estimator output, respectively, on specific time point and head
  - rezero g=-0.3
2. The value of mel-spectrogram
3. The attention map of "T", "I", "LL" and reference audio.(different)

## Question
1. The rezero g is fixed, good or not ?
2. 

## expected
1. If the target phoneme exist in reference, use reference phoneme, if not exist, use resnet result.