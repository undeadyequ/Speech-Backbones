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
def show_energy_contour(
        ref1, ref2,
        p2p_speech, fast_speech,
        ref2_p_start, ref2_p_end):
  """
  show energy contour of four (Hidden parts of ref2 out of ref2_p_start and ref2_p_end)
  """

def show_pitch_contour(ref1, ref2,
                       p2p_speech, fast_speech,
                       ref2_p_start, ref2_p_end):
  """
  show energy contour of four (Hidden parts of ref2 out of ref2_p_start and ref2_p_end)
  """



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
        {subtitle: data    # e.g. model : score_map}}
    rc_num:
    xy_label_sub:
    xtickslab:
    out_png:
    title:
    show_txt:
  Returns:
  """
  # fig size
  r_num, c_num = rc_num
  fig, axes = plt.subplots(r_num, c_num, figsize=(10 * (r_num / c_num), 10))
  plt.subplots_adjust(wspace=0.1, hspace=0.4)
  fontsize = 12

  n = 0
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

