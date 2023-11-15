# inference sample
fu
# interpolation test
python inference.py -f resources/filelists/synthesis.txt -c checkpts/grad-tts-libri-tts.pt -t 100 -s 12 -s2 10


# 
y_mask: (4, 1, 164)
y: (4, 80, 188)
mu_y: (4, 80, 188)
spk: (4, 64)
emo: (4, 768)



# sphinx
- add work directory for every import path in config.p\y
https://stackoverflow.com/questions/63957326/sphinx-does-not-show-class-instantiation-arguments-for-generic-classes-i-e-par



```python
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../hifi-gan'))

```

- tutorial
https://towardsdatascience.com/documenting-python-code-with-sphinx-554e1d6c4f6d



# other


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

