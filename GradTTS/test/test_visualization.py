import sys
sys.path.append('/Users/luoxuan/Project/tts/Speech-Backbones')


from GradTTS.exp.visualization import compare_pitch_contour, show_attn_map, search_syllable_index
import numpy as np


def test_show_attn_map():
    input = {
        "sub_data_dict" :{
            "happy": np.random.rand(25, 35),
            "angry": np.random.rand(25, 35),
            "sad": np.random.rand(25, 35),
            "surprise": np.random.rand(25, 35),
            "frustrated": np.random.rand(25, 35),
            },
        "rc_num": (2, 3),
        "xy_label_sub": ("phoneme", "phoneme"),
        "xtickslab": ["AHO"] * 25,
        "ytickslab": ["BDD"] * 35,
        "out_png": "out.png",
        "title": "attn_map"
        }
    show_attn_map(**input)   
    

def test_search_syllable_index():
    phoneme = [
        "",
        "HH",
        "",
        "IY1",
        "",
        11,
        "",
        "W",
        "",
        "AA1",
        "",
        "Z",
        "",
        11,
        "",
        "S",
        "",
        "T",
        "",
        "IH1",
        "",
        "S",
        "",
        "T",
        "",
        "IH1",
        "",
        "L",
        "",
        11,
        "",
        "IH0",
        "",
        "N",
        "",
        11,
        "",
        "DH",
        "",
        "AH0",
        "",
        11,
        "",
        "f",
        "",
        "o",
        "",
        "r",
        "",
        "e",
        "",
        "s",
        "",
        "t",
        "",
        "!",
        ""]

    syllabel_index = search_syllable_index(phoneme)
    syllabel_index_gd = [0, 5, 13, 20, 29, 35, 41]
    print(syllabel_index)



if __name__ == "__main__":
    test_search_syllable_index()