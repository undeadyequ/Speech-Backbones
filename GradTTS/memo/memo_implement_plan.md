- [Cross attention](#cross-attention)
- [Score function of gradTTS](#score-function-of-gradtts)
- [Quesition](#quesition)
  - [Condition and sample (Diffuser)](#condition-and-sample-diffuser)
  - [Condition and sample (Origin)](#condition-and-sample-origin)


## Cross attention
- K,V = Style
- Q = content


## Score function of gradTTS
- Linear attention is adopted in each layer
- Concatenation of input
  - t, x:  embed sepeartedly and concate.
  - spk, hidden, mu, xt: Directly concate to x
![unet_arch](../img/unet_arch.png)


## evaluation
- similarity compare picture
  - xticks: number/float -> phoneme
  - show xtickslabel
  - show title
  - show subtitle
  - show legend for each line

## Quesition
### Condition and sample (Diffuser)
- XT (stlye)
  - Sample
    - Sample -> sample channel 1
    - Spk    -> sample channel 2
  - Embedding
    - Emotion -> class embedding
    - Time    -> time embdding
    - PSD     -> add condition (image) or (mel-embedding)
  - Concate (addition_embed_type=image): 
    - **sample** = concate(sample, hint) where hint = self.add_embedding(image_embs, hint)
    - **emb** = embp(time) + embp(class_emb) + add_embedding(aug_emb) 
      - where embp = emb + time_project
  - Concate (addition_embed_type=text):
  - Question: 
    - Why seperate sample and embedding? what are they for? ->
    - Why t is needed in estimator?
    - During inference, which influenced the style more ? noised sample or class_embedding? 
- Hid (semantic)
  - Text
- model: down, mid, up
  - down (DownBlock2D + CrossAttnDownBlock2D):
    - IN: sample, emb
    - Downsample2D
      - ResnetBock: GroupNorm (AdaGroupNorm) -> LoRACompatibleConv -> LoRACompatibleLinear -> GroupNorm -> LoRACompatibleConv
        - AdaGroupNorm: **sample** = AdaGroupNorm(**sample**, **emb**)
    - CrossAttnDownBlock2D
      - crossAtt(k,v=**sample**, q=**txt**)   (sample=hidden_states, txt=encoder_hidden_states)
    - Quesition
      - Why emb(t) is separeted from x_t? <- homegenous data is processed in advance
      - Cross attention VS LinearAttention
  - mid: UNetMidBlock2DCrossAttn
    - IN: out of down

  - up: (CrossAttnUpBlock2D + UpBlock2D)
    - IN: out of mid

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
  - Concate: $\~{x_t} = concate(x_t, embp(spk), embp(emo), repeat(psd))$ where embp = emb + mlp
  - GAP:
    - Simple combination of style and content

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
    - ResnetBlock -> Residual+Rezero+LinearAttention -> ResnetBlock

  - up:
    - IN: out of mid
    - ResnetBock -> ResnetBlock -> Residual+Rezero+LinearAttention -> Upsample

```python
self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
```
