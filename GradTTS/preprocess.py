from GradTTS.text import text2sylstart_interp
from utils import parse_filelist, intersperse

def add_syl_start_on_phone(train_f):
    item_list = parse_filelist(train_f, split_char='|')
    syl_item_list = []
    for item in item_list:
        # item[0]: wav_path, item[1]: text, item[2]: phone, item[3]: emo
        text = item[3]
        # add syllable start
        syl_start = [str(s) for s in text2sylstart_interp(text)]
        item.append(",".join(syl_start))
        syl_item_list.append("|".join(item))
    # write to file
    train_syl_f = train_f[:-4] + "_syl.txt"
    with open(train_syl_f, 'w') as f:
        for item in syl_item_list:
            f.write(item + '\n')

if __name__ == '__main__':
    train_f = "/home/rosen/Project/FastSpeech2/preprocessed_data/ESD/val.txt"
    add_syl_start_on_phone(train_f)