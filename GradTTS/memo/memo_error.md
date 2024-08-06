
- No module named 'monotonic_align'
/home/rosen/anaconda3/envs/gradtts/bin/python /home/rosen/Project/Speech-Backbones/GradTTS/train_multi_speaker.py 
Traceback (most recent call last):
  File "/home/rosen/Project/Speech-Backbones/GradTTS/train_multi_speaker.py", line 16, in <module>
    import params
  File "/home/rosen/Project/Speech-Backbones/GradTTS/params.py", line 9, in <module>
    from model.utils import fix_len_compatibility
  File "/home/rosen/Project/Speech-Backbones/GradTTS/model/__init__.py", line 10, in <module>
    from .cond_tts import CondGradTTS
  File "/home/rosen/Project/Speech-Backbones/GradTTS/model/cond_tts.py", line 13, in <module>
    import monotonic_align
ModuleNotFoundError: No module named 'monotonic_align'
- Solution: copy that file from temparary folder


- 