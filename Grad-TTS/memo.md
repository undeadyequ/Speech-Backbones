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
