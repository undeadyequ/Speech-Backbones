

V2: adaptive melstyle
1. Align melstyle to the length that same as mu_y -> No Alignment
   - Reason: Better monotonic attention. it is not monotonic when size of two are different for not understand reason
   - Method: Frame position converted from phoneme position of mu_y
2. fix bugs
   - scale_dur.py:       set all <1 value to 0  -> <1 and >0
   - util_speech_cut.py: set torch.min(tensor)  ->  tensor[0]
   - set seq_dur start from 0 -> 1
Result
- crossAttn consentrate on the first phoneme

-> 

V3: Downsampling melstyle
