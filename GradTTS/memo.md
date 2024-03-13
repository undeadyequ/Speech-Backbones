# Problem
## Voice can not be synthesized by temp interpolation
  - self.to_out(context) has bias which is removed/not removed in inference and training stage.  <- context (b, d*l, s) is half masked on d*l dim in inference. -> modify it later
  - two residual  -> modify it later
  - rezero == -0.003?  -> why it is needed?
  - Layernorm didn't effect masked value (Still zero before/after norm)

# Implement
## Temp interpolation -> OK
## Freq interpolation -> On doing

```c
# sampleGuidence
loop: t < T
  if t <- tal:
    loop: i < epoch:
      x_guid <- x
      attn_score <- estimator(x_guided)
      minimize loss(attn_score)
  else:
    att_score <- estimator(x)
```

- sampleGuidence
  - hyper
    - epoch
    - tal
  - pitchMask generation
    - get pitch <- from input
    - get bins from freq of pitch?
      - mels -> freq_list -> bins  OK
      - mel pitch alignment?
        - check t in pitch extraction?
- Improvement
  - epoch for training x_guided in SasmpleGuidence?
  - melstyle is 1 length!

![layernorm.png](img%2Flayernorm.png)

# other
- pesudo algorithm test
[https://tex.stackexchange.com/questions/163768/write-pseudo-code-in-latex](pesudo algorithm)

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
- Close CLIPImageProjection
- tensorboard
- build motonotic

Traceback (most recent call last):
  File "/home/rosen/anaconda3/envs/gradtts/lib/python3.8/tkinter/__init__.py", line 4017, in __del__
    self.tk.call('image', 'delete', self.name)
RuntimeError: main thread is not in main loop
Exception ignored in: <function Image.__del__ at 0x7f6ac6053c10>
Traceback (most recent call last):
  File "/home/rosen/anaconda3/envs/gradtts/lib/python3.8/tkinter/__init__.py", line 4017, in __del__
    self.tk.call('image', 'delete', self.name)
RuntimeError: main thread is not in main loop


Exception ignored in: <function Image.__del__ at 0x7f6ac6053c10>
Traceback (most recent call last):
  File "/home/rosen/anaconda3/envs/gradtts/lib/python3.8/tkinter/__init__.py", line 4017, in __del__
    self.tk.call('image', 'delete', self.name)
RuntimeError: main thread is not in main loop
Tcl_AsyncDelete: async handler deleted by the wrong thread
Epoch: 11, iteration: 30925 | dur_loss: 0.4183737337589264, prior_loss: 1.5957911014556885, diff_loss: 0.030131766572594643:   0%|          | 6/3865 [04:41<50:20:30, 46.96s/it]
Traceback (most recent call last):
  File "/home/rosen/anaconda3/envs/gradtts/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1163, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
  File "/home/rosen/anaconda3/envs/gradtts/lib/python3.8/multiprocessing/queues.py", line 107, in get
    if not self._poll(timeout):
  File "/home/rosen/anaconda3/envs/gradtts/lib/python3.8/multiprocessing/connection.py", line 257, in poll
    return self._poll(timeout)
  File "/home/rosen/anaconda3/envs/gradtts/lib/python3.8/multiprocessing/connection.py", line 424, in _poll
    r = wait([self], timeout)
  File "/home/rosen/anaconda3/envs/gradtts/lib/python3.8/multiprocessing/connection.py", line 931, in wait
    ready = selector.select(timeout)
  File "/home/rosen/anaconda3/envs/gradtts/lib/python3.8/selectors.py", line 415, in select
    fd_event_list = self._selector.poll(timeout)
  File "/home/rosen/anaconda3/envs/gradtts/lib/python3.8/site-packages/torch/utils/data/_utils/signal_handling.py", line 66, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 6303) is killed by signal: Aborted. 

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/rosen/Project/Speech-Backbones/GradTTS/train_multi_speaker_lm.py", line 356, in <module>
    train_process_cond(configs)
  File "/home/rosen/Project/Speech-Backbones/GradTTS/train_multi_speaker_lm.py", line 226, in train_process_cond
    for batch in progress_bar:
  File "/home/rosen/anaconda3/envs/gradtts/lib/python3.8/site-packages/tqdm/std.py", line 1182, in __iter__
    for obj in iterable:
  File "/home/rosen/anaconda3/envs/gradtts/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 681, in __next__
    data = self._next_data()
  File "/home/rosen/anaconda3/envs/gradtts/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1359, in _next_data
    idx, data = self._get_data()
  File "/home/rosen/anaconda3/envs/gradtts/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1325, in _get_data
    success, data = self._try_get_data()
  File "/home/rosen/anaconda3/envs/gradtts/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1176, in _try_get_data
    raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e
RuntimeError: DataLoader worker (pid(s) 6303) exited unexpectedly
