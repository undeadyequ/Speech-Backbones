# Syntheized Problem
## Voice can not be synthesized by temp interpolation
  - self.to_out(context) has bias which is removed/not removed in inference and training stage.  <- context (b, d*l, s) is half masked on d*l dim in inference. -> modify it later
  - two residual  -> modify it later
  - rezero == -0.003?  -> why it is needed?
  - Layernorm didn't effect masked value (Still zero before/after norm)

# Temp interpolation -> OK
# Freq interpolation -> On doing
## Description
  - The generated mel-spectrogram is denoised by interpolation of two reference speech style on different frequence part including
    - The pitch related
    - The non-pitch related
  - This interpolation can
  
## Method
### Candiate1: Interp at different time
- The denoising process including **pitch-related/non-pitch-related denoising** in the order of timeline.
  - In pitch-related denoising, we denoise mel from a tuned initial noise which make the attention score more focus on pitch-relating freq rather than other part.
    - This tuned initial noise is trained by minimizing the sum of attention score which located in pitchMask. (and maximizing the sum of attention score located out of pitchMask with score scale)
  - In non-pitch-related denoising, we .... on non-pitch-relating freq rather than other part.
- **Attention score selection**: All attention
- Extention:
  - The concept of pitch-related/non-pitch-related can be expanded to **f04-related/non-f04_related denoising**
### Candidate2: Interp at same time ? <- may not be possibile
### Candidate3: Initiate two noise and 


- up and down
in_out: [(3, 64), (64, 128), (128, 256)]

orign_x_shape: [b, head, 80, d4]

input x_shape: [d1, d2, d3, d4]
d2: [(64, 128, 256), (128, 64)] -> [(h0, h0*2, h0*4), (h0*2, h0)]
d3: [(80, 40, 20), (20, 40)] -> [(d0, d0/2, d0/4), (d0/4, d0/2)] 
d4: [(l0, l0/2, l0/4), (l0/4, l0/2)]

- get attn_mask 
  - attn shape: (d * l , d * l_r)
  - input x shape: (b, h, d, l)
  - origin x shape: (b0, h0, d0, l0)

from (d1, d2, d3, d4)


up x shape: torch.Size([1, 64, 80, 228])
up x shape: torch.Size([1, 128, 40, 114])
up x shape: torch.Size([1, 256, 20, 57])
down x shape: torch.Size([1, 128, 20, 57])
down x shape: torch.Size([1, 64, 40, 114])


```python
## 1
if t < t_tal1:
  x_tune = x.clone()
  gama = 0.2  # the lower gama indicate more concentration on pitchMask
  x_tune, attn_score = estimater(x_tune)
  attn_sum_pitchMask_ref1 = attn_score[pitchMask==1]  # Sum of attention score in pitchMask
  attnSum_nonPitchMask_ref1 = attn_score[pitchMask==0] # Sum of attention score out of pitchMask
  
  loss = -attn_sum_pitchMask_ref1 + gama * attnSum_nonPitchMask_ref1
  loss.backward()

elif t_tal2 > t > t_tal1:
  loss = attn_sum_pitchMask_ref2 - gama * attnSum_nonPitchMask_ref2
else:
  pass
## 2
loss1 = -attn_sum_pitchMask_ref1 + gama * attnSum_nonPitchMask_ref1
loss2 = attn_sum_pitchMask_ref2 - gama * attnSum_nonPitchMask_ref2

## 3
loss = -pitchMask_ref1 - non_pitchMask_ref2(2nd noise)  <- is possible ?
```

## Implement
- hyper  -> OK
  - epoch
  - tal
- pitchMask generation -> OK
  - get pitch <- from input
  - get bins from freq of pitch?
    - mels -> freq_list -> bins  OK
    - mel pitch alignment?
      - check t in pitch extraction?
- set loss
  - ref code

## Problem
- The noise is not updated as expected
- pitchMask for each attention map

## Evaluation
- Pitch contour matching
- Non-pitch contour matching
  - energy
  - speaker
  - gender
  - age?

# other
- pesudo algorithm test
[https://tex.stackexchange.com/3questions/163768/write-pseudo-code-in-latex](pesudo algorithm)

- layer normalization
![layernorm.png](img%2Flayernorm.png)

# interpolation test
python inference.py -f resources/filelists/synthesis.txt -c checkpts/grad-tts-libri-tts.pt -t 100 -s 12 -s2 10


# Implement process
y_mask: (4, 1, 164)
y: (4, 80, 188)
mu_y: (4, 80, 188)
spk: (4, 64)
emo: (4, 768)

## Architect of x_T, z_T
- x_T = concat(z, mu) -> (b, 2, dim, l)
- z_T = concat(spk, melstyle, emolabel) -> (b, dim * 3, l)
 

## Unet process

## Implement CrossAttention
[](https://pytorch-forecasting.readthedocs.io/en/latest/_modules/pytorch_forecasting/models/temporal_fusion_transformer/sub_modules.html#ScaledDotProductAttention)
[](https://medium.com/@wangdk93/implement-self-attention-and-cross-attention-in-pytorch-1f1a366c9d4b)
https://cpp-learning.com/einops/#einops_-rearrange

- diffuser attention implement
  - AttnProcessor2_0


dims = [total_dim, *map(lambda m: dim * m, dim_mults)]  # [1, 64, 128, 256]
- dim = 64
- dim_mults = (1, 2, 4)

- in_out of resnet:
  [(1, 64), (64, 128), (128, 256)]
AssertionError: was expecting embedding dimension of 80, but got 40


## Estimator list
- Estimator List:


# other

## sphinx
- add work directory for every import path in config.p\y

https://stackoverflow.com/questions/63957326/sphinx-does-not-show-class-instantiation-arguments-for-generic-classes-i-e-par
```python
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../hifi-gan'))

```
- tutorial
https://towardsdatascience.com/documenting-python-code-with-sphinx-554e1d6c4f6d


# Error