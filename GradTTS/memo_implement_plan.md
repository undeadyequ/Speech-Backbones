1. Restore model
2. Model (X_T = u)
   1. Encoder
   2. Decoder
      - Score Estimator
        - Estimator(Energy/Pitch + Emotion Label) -> CrossNet: E(Q: K,V=())
        - 
3. load pitch, energy, duration
4. Model Paramater


## Cross attention
- K,V = Style
- Q = content


## Score function of gradTTS
- Linear attention is adopted in each layer
- Concatenation of input
  - t, x:  embed sepeartedly and concate.
  - spk, hidden, mu, xt: Directly concate to x
![unet_arch](img/unet_arch.png)



## Quesition
### Condition and sample (The reason)
- XT (stlye)
  - Sample
    - Sample -> sample channel 1
    - Spk    -> sample channel 2
  - Embedding
    - Emotion -> class embedding
    - Time    -> time embdding
    - PSD     -> add condition (image)
- Hid (semantic)
  - Text
- How XT and embedding combined? group normalization.
  - "DownBlock2D"
  - "CrossAttnDownBlock2D"
  - 

```python
self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
```

### Condition and sample (Origin)
- XT (stlye)
  - Sample
    - Sample -> sample channel 1
    - Spk    -> sample channel 2 (spk_emb)
    - emo    -> sample channel 3 (emo_emb)
  - Embedding
    - Emotion -> class embedding
    - Time    -> time embdding
    - PSD     -> add condition (image)
- Hid (semantic)
  - Text
- How XT and embedding combined? group normalization.
  - "DownBlock2D"
  - "CrossAttnDownBlock2D"
  - 

$
\~{x_t} = concate(x_t, embp(spk), embp(emo), repeat(psd))
$

where embp = emb + mlp

- model: down, mid, up
  - down:
    - IN: $\~{x_t}, emb(t)$
    - ResnetBock -> ResnetBlock -> Residual+Rezero+LinearAttention -> Downsample
      - ResnetBock: Block(Block($\~{x_t}$) + mlp(emb(t)))
        - Block: Conv2d -> GroupNorm -> Mish
          - GroupNorm: Norm on a group with similar channels.
          - mish: An activation with suppression in negative value and same with relu in positive value
      - LinearAttention: attention with k,q,v == x 
  - Quesition
    - Why emb(t) is separeted from x_t? <- homegenous data is processed in advance
    - Cross attention VS LinearAttention
  
  - mid:
    - IN: out of down
    - mid\_block1 -> mid\_block2$

  - up:
    - IN: out of mid
    - resnet1 -> resnet2 -> attn -> upsample


```python
self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
```
