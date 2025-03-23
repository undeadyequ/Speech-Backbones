from cProfile import label
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import json
#import pandas as pd
import math
import collections
import numpy as np
from matplotlib.pyplot import title
## show psd

def vis_psd(pitch_dict_for_vis, prosody_dict, out_png):
    rc_num = (3, 2)
    legend_mark = ("*", "v", ".")
    xy_label_sub = ("frame", "Hz")

    txt_n = len(prosody_dict["Angry"]["reference"]["phonemes"])
    for txt_id in range(txt_n):
        # vis_speechid = 1   # 1: parallel data, 0~: Non-parallel data
        phonems = prosody_dict["Angry"]["reference"]["phonemes"][txt_id]
        show_txt = True
        title_extra = f"txt{id}"
        """
        if show_txt_id == 0:
            show_txt = True
            title_extra = "Text1"
        else:
            show_txt = False
            title_extra = "Parallel task"
        """
        print("3.1. do pitch comaration visualization on {}".format(" ".join(
            prosody_dict["Angry"]["reference"]["phonemes"][txt_id])))

        ####### Pitch Contour ############
        compare_pitch_contour(
            pitch_dict_for_vis,
            rc_num,
            legend_mark=legend_mark,
            xy_label_sub=xy_label_sub,
            xtickslab=phonems,
            out_png=out_png.format("pitch", title_extra.split(" ")[0]),
            title="Pitch contour of speech synthesized on reference, emoMix, proposed ({})".format(title_extra),
            show_txt=show_txt)

        print("5. do energy comaration visualization on {}st txt".format(txt_id))
        energy_dict_for_vis = dict()  # {"emo": {"model1": []}}
        for emo, m_psd in prosody_dict.items():
            energy_dict_for_vis[emo] = {}
            for model, psd in m_psd.items():
                if model not in energy_dict_for_vis[emo].keys():
                    energy_dict_for_vis[emo][model] = []
                if model != "reference":
                    energy_dict_for_vis[emo][model] = psd["energy"][txt_id]
                else:
                    energy_dict_for_vis[emo][model] = psd["energy"][0]

        ####### Energy Contour ############
        compare_pitch_contour(
            energy_dict_for_vis,
            rc_num,
            legend_mark=legend_mark,
            xy_label_sub=xy_label_sub,
            xtickslab=phonems,
            out_png=out_png.format("energy", title_extra.split(" ")[0]),
            title="Energy contour of speech synthesized on reference, emoMix, proposed ({})".format(title_extra),
            show_txt=show_txt
        )

def vis_crossAttn(attn_map_for_vis, prosody_dict, out_png, show_txt=0):

    # attn_map_for_vis -> {"model1": {"emo1": [np.array([syn_frames, ref_frames]), durations, phonemes] }}}
    for model_n, emo_attn_dict in attn_map_for_vis.items():  # {"model1": {"emo1": np.array([syn_frames, ref_frames])}}}
        rc_num = (3, 2)
        xy_label_sub = ("phoneme", "phoneme")
        title_extra = "non_parallel"
        t1, b1, h1 = 0, 1, 0

        syn_phonems = prosody_dict["Angry"][model_n]["phonemes"][show_txt]
        ref_phonemes = prosody_dict["Angry"]["reference"]["phonemes"][show_txt]

        show_attn_map(
            emo_attn_dict,
            rc_num,
            xy_label_sub=xy_label_sub,
            xtickslab=ref_phonemes,
            ytickslab=syn_phonems,
            out_png=out_png.format(model_n, "_t_{}_b_{}_h_{}".format(t1, b1, h1)),
            # "result/{}_similarity_{}.png"
            title="Attention map in crossAttention after masking ({})".format(title_extra)
        )


## compare pitch contour

def compare_pitch_contour(
        sub_leg_data_dict,
        rc_num,
        legend_mark,
        xy_label_sub,
        xtickslab,
        out_png,
        title="pitch contour",
        show_txt=True
):
  """
  args:
    sub_leg_data_dict:
    {subtitle:{         # e.g. emotion
        legend: data    # e.g. model # e.g. pitch
    }}
    rc_num: (2, 3)

    title: ""
    legend_mark: ["*", "o"]  # same len as len(subtitle.keys())  -> assert
    xy_label_sub: ("xlabel", "ylabel")
  Returns:

  """
  # fig size
  r_num, c_num = rc_num
  fig, axes = plt.subplots(r_num, c_num, figsize=(10 * (r_num / c_num), 10))
  plt.subplots_adjust(wspace=0.1, hspace=0.4)
  fontsize = 12

  n = 0
  for i, (sub, leg_data) in enumerate(sub_leg_data_dict.items()):
    r = int(n / c_num)
    c = int(n % c_num)
    axes[r, c].set_title(sub, fontsize=fontsize)

    axes[r, c].set_xticks(range(len(xtickslab)))
    #ax.set_xticklabels(farmers)
    #xtickslab_new = "a a b b c c a a b b c c d e f g j o k l"
    #axes[r, c].set_xticklabels(xtickslab.split(" "), rotation=60, fontsize=fontsize, minor=True)
    axes[r, c].set_xticklabels(xtickslab)

    for j, (legend, data) in enumerate(leg_data.items()):
      x_value = range(len(data))
      if legend == "gradtts":
        legend = "EmoMix"
      elif legend == "gradtts_cross":
        legend = "Proposed"
      axes[r, c].plot(x_value, data, label=legend, marker=legend_mark[j])
      axes[r, c].set_xlabel(xy_label_sub[0])
      axes[r, c].set_ylabel(xy_label_sub[1])

      # axes[r, c].set_xlim(right=xticks_columns[psd])

      # show reference phoneme on non-parallel style

      if show_txt and legend == "reference":
        axes[r, c].text(x_value[0], data[0], "iː w ɐ z s tʲ ɪ ɫ ɪ n d̪ ə f ɒ ɹ ɪ s t", ha='left', rotation=5, wrap=True, c="green")

    if r == 0 and c == 0:
      axes[r, c].legend(loc="upper left", ncol=3, fontsize=fontsize, bbox_to_anchor=(-0.1, 1, 1.2, 0.2))
    n += 1

  fig.suptitle(title)
  fig.savefig(out_png, dpi=300)


def show_attn_map(
        sub_data_dict,
        rc_num,
        xy_label_sub,
        xtickslab,
        ytickslab,
        out_png,
        title="attn_map"
):
    """
    Args:
    sub_data_dict:
        {model: data}}
    rc_num:
    xy_label_sub:
    xtickslab:
    out_png:
    title:
    show_txt:
    Returns:
    """
    # fig size
    #


    r_num, c_num = rc_num
    fig, axes = plt.subplots(r_num, c_num, figsize=(10 * (r_num / c_num), 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    fontsize = 12

    n = 0

    """
    # attn_map_for_vis -> {"model1": {"emo1": [np.array([syn_frames, ref_frames]), durations, phonemes] }}}
    for i, (emo, syn_attn_dur_phone, ref_dur_phoneme) in enumerate(sub_data_dict.items()):
        
        attn, syn_durs, syn_phones = syn_attn_dur_phone[0:2]  
        ref_durs, ref_phones = ref_dur_phoneme[0:1]
        syn_durs_inc = increase_sum_of_previous(syn_durs_inc)
        ref_durs_inc = increase_sum_of_previous(syn_durs_inc)
        axes[r, c].set_xlabel(syn_durs_inc, labels=syn_phones, fontsize=?)
        axes[r, c].set_xlabel(ref_durs_inc, labels=phones, fontsize=?)
        
        
        for ax_x in syn_durs_inc:
            ax.axvline(x=ax_x, color="blue", linestyle="--")
        for ax_y in ref_durs_inc:
            ax.axvline(y=ax_y, color="black", linestyle="--")
    """


    for i, (sub, data) in enumerate(sub_data_dict.items()):


        r = int(n / c_num)
        c = int(n % c_num)

        axes[r, c].set_title(sub, fontsize=fontsize)

        axes[r, c].set_xticks(range(len(xtickslab)))
        axes[r, c].set_xticklabels(xtickslab)
        axes[r, c].set_yticklabels(ytickslab)

        pc = axes[r, c].pcolor(data, cmap=plt.cm.Blues, alpha=0.9)
        axes[r, c].set_xlabel(xy_label_sub[0])
        axes[r, c].set_ylabel(xy_label_sub[1])
        fig.colorbar(pc, ax=axes[r, c])
        n += 1
    fig.suptitle(title)
    fig.savefig(out_png, dpi=300)

