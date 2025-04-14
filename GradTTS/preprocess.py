
from GradTTS.text import extend_phone2syl
from utils import parse_filelist, intersperse

def add_syl_start(train_f):
    # read
    filelist = parse_filelist(train_f, split_char='|')
    for index in range(len(filelist)):
        basename, speaker, phone, text = filelist[index][0:4]

