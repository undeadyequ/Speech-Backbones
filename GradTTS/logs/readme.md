# models for evaluation (model_config_input_extra)
## model -> BoneModel
- stdit
- stditCross: self + cross + FFT
- stditMocha:
- unet: unet model
- dit: DiT model
## config -> small change
- base
- qkNorm
- monoHard: hard monotonic attention mask
- guideLoss: guide crossAttention by monotonic matrix
- phonemeRoPE: phoneme-level positional Embedding
- refEncoder: mlp/vae/vaeGRL
## input -> ref
- wav2vec2: or melstyle
- pCodec: prosody
- melVae: vae of melspectrogram
- melVaeGRL: vae of melspectrogram with GRL to phoneme
## extra
- noGlobal


# Implement plan

model.yaml


interpEmoTTS_frame2binAttn_noJoint
other
styleEnhancedTTS
styleEnhancedTTS_dit
styleEnhancedTTS_stdit
styleEnhancedTTS_stdit_chmp
styleEnhancedTTS_stditMocha
styleEnhancedTTS_stditMocha_guideloss_codec
styleEnhancedTTS_stditMocha_noguideloss_codec
styleEnhancedTTS_stditMocha_norm
styleEnhancedTTS_stditMocha_norm_guideloss_vqvae
styleEnhancedTTS_stditMocha_norm_guidloss
- add guidloss

styleEnhancedTTS_stditMocha_norm_hardMonMask
styleEnhancedTTS_stditMocha_norm_hardMonMask_psd
styleEnhancedTTS_stditMocha_normTest
styleEnhancedTTS_stditTest
unkown


epoch = 145

|                                   | Quality | LocalStyleEnhance | MonoAttn                 | Unparal  | Other                                          |
| --------------------------------- | ------- | ----------------- | ------------------------ | -------- | ---------------------------------------------- |
| *_stditMocha_norm_hardMonMask     | 3-5     | 3-5               | MonoLine                 | Non-Entg | MonoAttn is white for some samples (epoch>155) |
| *_stditMocha_norm_guidloss        | 5       | 3?                | GuitarStrings (ecept e1) | Entg     |                                                |
| *_stditMocha_norm_hardMonMask     | 3-5     | 3-5               | MonoLine                 | Non-Entg |                                                |
| *_stditMocha_norm_guideloss_vqvae | 3       | 3-5               | mulitMonoLine            |          |                                                |
| *_guideloss_codec                 | 2       | 1                 | MonoLine                 | Non-Entg |                                                |
| *_noguideloss_codec               | 2       | 3                 | GuitarStrings            | Non-Entg |                                                |



# Result
- Quality is bad when when given vqvae and codec, while it is good when given melstyle


# quesiton
- Why monoLine, but not monoLine with chunk
- Why Quality is bad -> use classifier-free guidence law (p=0.3)
- Why GuitarStrings attention is good at localstyleEnhance


# Improvement
- model_config_input_other
  - stdit_base_codec_pSet
  - stdit_lossguide_codec_pSet
- Copy key result to keyResult after training 
- Evaluate parallel and unparllel task
- change emo1 -> angry
- sampleNameing: 
  - spk19_ang_para1_epoch165
  - spk19_hap_unpara1_epoch165

```c
cp samples/*160* /home/rosen/Project/Speech-Backbones/GradTTS/logs/styleEnhancedTTS_stditMocha_guideloss_codec/samples_epoch160
cp img/cross*160* /home/rosen/Project/Speech-Backbones/GradTTS/logs/styleEnhancedTTS_stditMocha_guideloss_codec/samples_epoch160
``` 

## Other
- pSet: classifier-free training set (mix traning cond/uncond with p)
- 


