from cProfile import label
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
from GradTTS.exp.utils import convert_xydur_xybox
from GradTTS.text import extend_phone2syl


def vis_psd(pitch_dict_for_vis, energy_dict_for_vis, prosody_dict, out_png, txt_id, show_ref_phone_on_line=True):
    rc_num = (3, 2)
    legend_mark = ("*", "v", ".")  # reference, guide, ropePhone_guide
    legend_color = ("grey", "lightblue", "blue")
    legend_linestyle = ("dashed", "solid", "solid")
    xy_label_sub = ("frame", "Hz")


    #txt_n = len(prosody_dict["Angry"]["reference"]["phonemes"])   # should have ten
    syn_phones = prosody_dict["Angry"]["reference"]["phonemes"][txt_id]   # syn_phones should be same for all emotion

    title_extra = f"txt{txt_id}"
    print("3.1. do pitch comaration visualization on {}".format(" ".join(
        prosody_dict["Angry"]["reference"]["phonemes"][txt_id])))

    ####### Pitch Contour ############
    compare_pitch_contour(
        pitch_dict_for_vis,
        rc_num,
        legend_mark=legend_mark,
        legend_color=legend_color,
        legend_linestyle=legend_linestyle,
        xy_label_sub=xy_label_sub,
        xtickslab=syn_phones,
        out_png=out_png.format("pitch", title_extra.split(" ")[0]),
        title="Pitch contour of speech synthesized on reference, emoMix, proposed ({})".format(title_extra),
        show_txt=show_ref_phone_on_line)
    print("5. do energy comaration visualization on {}st txt".format(txt_id))


    ####### Energy Contour ############
    compare_pitch_contour(
        energy_dict_for_vis,
        rc_num,
        legend_mark=legend_mark,
        legend_color=legend_color,
        legend_linestyle=legend_linestyle,
        xy_label_sub=xy_label_sub,
        xtickslab=syn_phones,
        out_png=out_png.format("energy", title_extra.split(" ")[0]),
        title="Energy contour of speech synthesized on reference, emoMix, proposed ({})".format(title_extra),
        show_txt=show_ref_phone_on_line
    )

def vis_crossAttn(attn_map_for_vis, prosody_dict, out_png, show_txt=0):
    # attn_map_for_vis -> {"model1": {"emo1": [np.array([syn_frames, ref_frames]), durations, phonemes] }}}
    for model_n, emo_attn_dict in attn_map_for_vis.items():  # {"model1": {"emo1": np.array([syn_frames, ref_frames])}}}
        rc_num = (3, 2)
        xy_label_sub = ("phoneme", "phoneme")
        title_extra = "non_parallel"
        t1, b1, h1 = 0, 1, 0

        #syn_phonems = prosody_dict["Angry"][model_n]["phonemes"][show_txt]
        #ref_phonemes = prosody_dict["Angry"]["reference"]["phonemes"][show_txt]
        syn_phonems = prosody_dict["Angry"][model_n]["phonemes"][show_txt]
        ref_phonemes = prosody_dict["Angry"][model_n]["phonemes"][show_txt]
        show_attn_map(
            emo_attn_dict,
            rc_num,
            xy_label_sub=xy_label_sub,
            xtickslab=ref_phonemes,
            ytickslab=syn_phonems,
            out_png=out_png.format(model_n, "_t_{}_b_{}_h_{}".format(t1, b1, h1)),
            # "result/{}_similarity_{}.png"
            title="Attention map in crossAttention after masking ({})".format(title_extra),
            tick_gran="syllable"
        )


## compare pitch contour
def compare_pitch_contour(
        sub_leg_data_dict,
        rc_num,
        legend_mark,
        legend_color,
        legend_linestyle,
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
      axes[r, c].plot(x_value, data, label=legend, marker=legend_mark[j], color=legend_color[j], linestyle=legend_linestyle[j])
      axes[r, c].set_xlabel(xy_label_sub[0])
      axes[r, c].set_ylabel(xy_label_sub[1])
      # axes[r, c].set_xlim(right=xticks_columns[psd])

      # show reference phoneme for non-parallel style
      if show_txt and legend == "reference":
        axes[r, c].text(x_value[0], data[0], "iː w ɐ z s tʲ ɪ ɫ ɪ n d̪ ə f ɒ ɹ ɪ s t",
                        ha='left', rotation=5, wrap=True, c="green")
    if r == 0 and c == 0:
      axes[r, c].legend(loc="upper left", ncol=3, fontsize=fontsize, bbox_to_anchor=(-0.1, 1, 1.2, 0.2))
    n += 1
  fig.delaxes(axes[2, 1])
  fig.suptitle(title)
  fig.savefig(out_png, dpi=300)

def show_attn_map(
        sub_data_dict,
        rc_num,
        xy_label_sub,
        xtickslab,
        ytickslab,
        out_png,
        title="attn_map",
        tick_gran="phoneme",
        need_auxline=False
):
    """
    Args:
    sub_data_dict:
        {model: data}}
    """
    r_num, c_num = rc_num
    fig, axes = plt.subplots(r_num, c_num, figsize=(10 * (r_num / c_num), 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    fontsize = 12
    n = 0

    # attn_map_for_vis -> {"model1": {"emo1": [np.array([syn_frames, ref_frames]), durations, phonemes] }}}
    for emo, contents in sub_data_dict.items():
        r = int(n / c_num)
        c = int(n % c_num)
        axes[r, c].set_title(emo, fontsize=fontsize)

        attn, syn_phone_durs, syn_phones, ref_phone_durs, ref_phones = contents
        pc = axes[r, c].pcolor(attn, cmap=plt.cm.Blues, alpha=0.9)

        # change gran
        if tick_gran == "syllable":
            syn_syls, syn_syl_durs = extend_phone2syl(syn_phones, syn_phone_durs)
            ref_syls, ref_syl_durs = extend_phone2syl(ref_phones, ref_phone_durs)
            syn_labels, ref_labels = syn_syls, ref_syls
            syn_durs, ref_durs = syn_syl_durs, ref_syl_durs
        else:
            syn_labels, ref_labels = syn_phones, ref_phones
            syn_durs, ref_durs = syn_phone_durs, ref_phone_durs

        # set label
        syn_durs_inc, ref_durs_inc = [0], [0]
        syn_durs_inc.extend([sum(syn_durs[:i + 1]) for i in range(len(syn_durs))])
        ref_durs_inc.extend([sum(ref_durs[:i + 1]) for i in range(len(ref_durs))])
        syn_labels.append("")  # to align label to the left
        ref_labels.append("")  # to align label to the left

        #print("syn_durs_inc, syn_labels: {} {}".format(syn_durs_inc, syn_labels))
        #print("ref_durs_inc, ref_labels: {} {}".format(ref_durs_inc, ref_labels))

        if len(syn_durs_inc) != len(syn_labels) or len(ref_durs_inc) != len(ref_labels):
            IOError("durs and phones should have same lens {} {} {} {}".format(
                len(syn_durs_inc), len(syn_labels), len(ref_durs_inc), len(ref_labels)))
        axes[r, c].set_xticks(syn_durs_inc)
        axes[r, c].set_xticklabels(syn_labels, fontsize=8, rotation=45, ha="left")
        axes[r, c].set_yticks(ref_durs_inc)
        axes[r, c].set_yticklabels(labels=ref_labels, fontsize=8, rotation=45, va="center")

        # set auxiliary lines
        if need_auxline:
            for ax_x in syn_durs_inc:
                axes[r, c].axvline(x=ax_x, color="blue", linestyle="--", linewidth=0.3)
            for ax_y in ref_durs_inc:
                axes[r, c].axhline(y=ax_y, color="blue", linestyle="--", linewidth=0.3)

        # set rectangle
        xywz_list = convert_xydur_xybox(syn_durs_inc, ref_durs_inc)
        for x, y, w, h in xywz_list:
            axes[r, c].add_patch(plt.Rectangle((x, y), w, h, ls="-", ec="red", fc="none", linewidth=0.5))

        fig.colorbar(pc, ax=axes[r, c])
        n += 1

    fig.suptitle(title)
    fig.delaxes(axes[2, 1])
    fig.savefig(out_png, dpi=300)

def show_attn_map_bk(
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
    for i, (emo, syn_attn_dur_phone, ref_dur_phoneme) in enumerate(sub_data_dict.rm_items()):
        
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

        #axes[r, c].set_xticks(range(len(xtickslab)))
        #axes[r, c].set_yticks(range(len(ytickslab)))
        axes[r, c].set_xticklabels(xtickslab)
        axes[r, c].set_yticklabels(ytickslab)

        pc = axes[r, c].pcolor(data, cmap=plt.cm.Blues, alpha=0.9)
        axes[r, c].set_xlabel(xy_label_sub[0])
        axes[r, c].set_ylabel(xy_label_sub[1])
        fig.colorbar(pc, ax=axes[r, c])
        n += 1
    fig.suptitle(title)
    fig.savefig(out_png, dpi=300)

