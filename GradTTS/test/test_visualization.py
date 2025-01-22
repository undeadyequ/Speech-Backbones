import sys
sys.path.append('/Users/luoxuan/Project/tts/Speech-Backbones')


from GradTTS.exp.visualization import show_pitch_contour, show_energy_contour, compare_pitch_contour, show_attn_map
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
    
    
if __name__ == "__main__":
    test_show_attn_map()