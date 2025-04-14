
# Statisitic
1. Filter 

Reference/Text=10, Emo=5, Spk=2 (Para: same, unpara: different) 


 - Filter train.txt by given spk starting, text ending (text should be very emotional).  -> How about punctuation (remove it because emotion is convey by psd not punctuation) -> check how to remoe punctuation in fastspeech preprocessing
- Para: mcd, pitch (Get MCD. Get Pitch from p-lelel MTA/extraction)
 - Vowel tend to be low pitch, while consonants (voiced) tend to high pitch.
- unpara: pitch -> syllabel-level pitch diff


```python
#cat train.txt | grep -E "\b0019" <- start with 0019 by extended 
grep -x -f spkr.txt train.txt | train_spkr.txt
grep -x -f text.txt train_spkr.txt | train_spkr_text.txt
```


Pitch contour
- ref: black_dash, base: green, mono-guide: light blue, mono-guide+rope_syl: dark blue
- Use dur/phone got from model, Phoneme-level contour ->
- Need to choose better example


Re-train rope_sel model
- give syllable start list in seq_dur
 - Old: [2, 3, 2, 5, 2] -> [1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5]
 - New: [2, 3, 2, 5, 2]  + [0, 5, 12] -> [1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 4, 4] (use syllable length instead of phonme length)


Enhancement (only for para task)
- Make the attention consentrat on the target style
- why not just improve posd
$
attnEnhance[i, j:j+M] = max(attn[i, j:j+M] + sum(attn[i, !(j:j+M)]) / M * con_score, 1)


attnEnhance[i, !(j:j+M)] = attn[i, !(j:j+M)] - (sum(attnEnhance[i, :]) - 1) / (N - M)   # Exceed part is reduced averagely in other phoneme 
$


N is number of reference frames
M is number of frames of specific phoneme
cons_score: consentration score (0 ~ 1)
- Exp: I(2) L(5) AI(3) K(2), N = 2 + 5 + 3 + 2, M = 3


```python


def enhancePhoneAttention(attn, ref_ind_dur, syn_ind_dur, enh_score=1.0):
 """
 attn: (b, h, ref_len, syn_len)
 enh_score: (0, 1)  -> Currently can not set to 1.0 because minus problem
 """
 ehance_type = "simple"
 i, N = syn_ind_dur
 j, M = ref_ind_dur
 for n in range(N):
   p_exp = [ind for ind in range(attn.shape[2]) if ind not in range(j, j + M)]
   if enhance_type == "simple":
     attn[i + n, j:j+m] = torch.clamp(attn[i + n, j:j+m] + 0.5, max=1.0)
     attn[i + n, p_exp] = torch.clamp(attn[i, p_exp] - 0.5, min=0.0)
   else:
     attn[i + n, j:j+m] = max(attn[i + n, p_exp] + sum(attn[i + n, p_exp]) / M * enh_score, 1)
     attn[i + n, p_exp] = attn[i, p_exp] - (sum(attnEnhance[i, :]) - 1) / (N - M)   # ???what about minus value after substraction???


   #attn[i + n, p_exp] = 0


```






4/9






# Catchup tech


| Model      | Release | Speaker | Emotion | Train_code | infer_code |
| ---------- | ------- | ------- | ------- | ---------- | ---------- |
| SparkTTS   | 2025/   |         |         |            |            |
| F5-TTS     |         |         |         |            |            |
| Zonos      |         |         |         |            |            |
| MegaTTS3   |         |         |         |            |            |
| GPT-SoVITS |         |         |         |            |            |
|            |         |         |         |            |            |

