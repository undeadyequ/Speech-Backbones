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

```python
## Score
score_emo = self.estimator(xt, mask, mu, t, spk, hidden_stats1)

## -> self.estimator
x = torch.stack([mu, x, s, emo], 1) 
for resnet1, resnet2, attn, downsample in self.downs:
   mask_down = masks[-1]
   x = resnet1(x, mask_down, t)
   x = resnet2(x, mask_down, t)
   x = attn(x)
   hiddens.append(x)
   x = downsample(x * mask_down)
   masks.append(mask_down[:, :, :, ::2])

## -> self.downs
self.downs.append(torch.nn.ModuleList([
            ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
            ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
            Residual(Rezero(LinearAttention(dim_out))),
            Downsample(dim_out) if not is_last else torch.nn.Identity()]))

## ->(1) ResnetBlock (block)
self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                      dim_out))
self.block1 = Block(dim, dim_out, groups=groups)
def forward(self, x, mask, time_emb):
   h = self.block1(x, mask)
   h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
   h = self.block2(h, mask)
   output = h + self.res_conv(x * mask)
   return output

### -> Block
self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                 padding=1), torch.nn.GroupNorm(
                                 groups, dim_out), Mish())
def forward(self, x, mask):
   output = self.block(x * mask)
   return output * mask

## ->(2) LinearAttention
self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)     
def forward(self, x):
   b, c, h, w = x.shape
   qkv = self.to_qkv(x)
   q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                        heads = self.heads, qkv=3)            
   k = k.softmax(dim=-1)
   context = torch.einsum('bhdn,bhen->bhde', k, v)
   out = torch.einsum('bhde,bhdn->bhen', context, q)
   out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                  heads=self.heads, h=h, w=w)
   return self.to_out(out)

## ->(3) CrossAttention -> Transformer2DModel? nn.multiheadattention
self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
def forward(self, q=x, kv=encoder_hidden_stats=?):
   attn_output, attn_output_weights = multihead_attn(query, key, value)

```

## Score function of diffuser
```python
unet = UNet2DConditionModel(
   block_out_channels=(32, 64),
   layers_per_block=2,
   sample_size=32,
   in_channels=4,
   out_channels=4,
   down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
   up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
   cross_attention_dim=32,
)

## 1. time
timesteps = timesteps.expand(sample.shape[0])
t_emb = self.time_proj(timesteps)
emb = self.time_embedding(t_emb, timestep_cond)
emb = torch.cat([emb, class_emb], dim=-1)

## 2. pre-process
sample = self.conv_in(sample)


## 3. down
for downsample_block in self.down_blocks:
   sample, res_samples = downsample_block(
      hidden_states=sample,
      temb=emb,
      encoder_hidden_states=encoder_hidden_states,
      attention_mask=attention_mask,
      cross_attention_kwargs=cross_attention_kwargs,
      encoder_attention_mask=encoder_attention_mask,
      **additional_residuals,
   )

## -> self.down_blocks
down_block = get_down_block(
      down_block_type,
      num_layers=layers_per_block[i],
      transformer_layers_per_block=transformer_layers_per_block[i],
      in_channels=input_channel,
      out_channels=output_channel,
      temb_channels=blocks_time_embed_dim,
      add_downsample=not is_final_block,
      resnet_eps=norm_eps,
      resnet_act_fn=act_fn,
      resnet_groups=norm_num_groups,
      cross_attention_dim=cross_attention_dim[i],
      num_attention_heads=num_attention_heads[i],
      downsample_padding=downsample_padding,
      dual_cross_attention=dual_cross_attention,
      use_linear_projection=use_linear_projection,
      only_cross_attention=only_cross_attention[i],
      upcast_attention=upcast_attention,
      resnet_time_scale_shift=resnet_time_scale_shift,
      attention_type=attention_type,
      resnet_skip_time_act=resnet_skip_time_act,
      resnet_out_scale_factor=resnet_out_scale_factor,
      cross_attention_norm=cross_attention_norm,
      attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
      dropout=dropout,
)
self.down_blocks.append(down_block)

## <- get_down_block
class CrossAttnDownBlock2D(nn.Module):
   def forward(
      hidden_states: torch.FloatTensor,
      temb: Optional[torch.FloatTensor] = None,
      encoder_hidden_states: Optional[torch.FloatTensor] = None,
      attention_mask: Optional[torch.FloatTensor] = None,
      cross_attention_kwargs: Optional[Dict[str, Any]] = None,
      encoder_attention_mask: Optional[torch.FloatTensor] = None,
      additional_residuals=None,
   ):
      ...
      blocks = list(zip(self.resnets, self.attentions))
      for i, (resnet, attn) in enumerate(blocks):
         hidden_states = resnet(hidden_states, temb, scale=lora_scale)
         hidden_states = attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
         )[0]
      output_states = output_states + (hidden_states,)

      ## ->(1) resnets, attentions
      for i in range(num_layers):
         resnets.append(
               ResnetBlock2D(
                  in_channels=in_channels,
                  out_channels=out_channels,
                  temb_channels=temb_channels,
                  eps=resnet_eps,
                  groups=resnet_groups,
                  dropout=dropout,
                  time_embedding_norm=resnet_time_scale_shift,
                  non_linearity=resnet_act_fn,
                  output_scale_factor=output_scale_factor,
                  pre_norm=resnet_pre_norm,
               )
         )
         attentions.append(
            Transformer2DModel(
               num_attention_heads,
               out_channels // num_attention_heads,
               in_channels=out_channels,
               num_layers=transformer_layers_per_block,
               cross_attention_dim=cross_attention_dim,
               norm_num_groups=resnet_groups,
               use_linear_projection=use_linear_projection,
               only_cross_attention=only_cross_attention,
               upcast_attention=upcast_attention,
               attention_type=attention_type,
            )
         )

```




