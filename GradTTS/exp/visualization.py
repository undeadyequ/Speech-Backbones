from cProfile import label
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import json
import pandas as pd
import math
import collections
import numpy as np
from Cython.Compiler.Parsing import p_arith_expr


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
        rc_num,
        pitch_pair_sub,
        title,
        subtitle,
        pair_legend,
        pari_mark,
        xy_label_sub
):
  """
  args:
    rc_num: (2, 3)
    pitch_pair_sub: [([p1_0,..], [p2_0,..]), []]
    title: ""
    subtitle: ["ang", "hap", ...]
    pair_legend: ["gradTTS", "InterpTTS"]
    pari_mark: ["*", "o"]
    xy_label_sub: [("xlabel", "ylabel"), ()]
  Returns:

  """
  fig, axes = plt.subplots(row_n, col_n, figsize=(10 * ratios, 10))
  plt.subplots_adjust(wspace=0.1, hspace=0.1)

  r_num, c_num = rc_num

  for c in range(c_num):
    for r in range(r_num):
      sub_num = c * c_num + r
      p0 = pitch_pair_sub[sub_num][0]
      p1 = pitch_pair_sub[sub_num][1]

      p0_legend = pair_legend[0]
      p1_legend = pair_legend[1]

      assert len(p1) == len(p0)
      x_value = range(len(p1))
      axes[r, c].plot(x_value, p0, label=p0_legend, marker=pair_legend[0])
      axes[r, c].plot(x_value, p1, label=p1_legend, marker=pair_legend[0])

      axes[r, c].set_xlabel(xy_label_sub[sub_num][0])
      axes[r, c].set_ylabel(xy_label_sub[sub_num][1])


      if False:
        #axes[r, c].set_xlim(right=xticks_columns[psd])
        #axes[r, c].set_xticks(x_frame_for_xticks)
        #axes[r, c].set_xticklabels(show_xticketlabel.split(" "), rotation=60, fontsize=font_size,
                                   #minor=minor)