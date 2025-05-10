from cProfile import label
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
from GradTTS.exp.utils import convert_xydur_xybox, clean_phone
from GradTTS.text import extend_phone2syl

def vis_psd(pitch_dict_for_vis, energy_dict_for_vis, prosody_dict, out_png, txt_id, show_ref_phone_on_line=True):
    rc_num = (3, 2)
    legend_mark = ("*", "v", ".")  # reference, guide, ropePhone_guide
    legend_color = ("grey", "lightblue", "blue")
    legend_linestyle = ("dashed", "solid", "solid")
    xy_label_sub = ("frame", "Hz")

    #txt_n = len(prosody_dict["Angry"]["reference"]["phonemes"])   # should have ten
    real_text_id = max([i if txt_name[-1] == str(txt_id) else -1 for i, txt_name in
                        enumerate(prosody_dict["Angry"]["reference"]["speechid"])])  # psd["pitch"] is not sorted

    syn_phones = prosody_dict["Angry"]["reference"]["phonemes"][real_text_id]   # syn_phones should be same for all emotion

    title_extra = f"txt{txt_id}"
    print("3.1. do pitch comaration visualization on {}".format(" ".join(
        prosody_dict["Angry"]["reference"]["phonemes"][txt_id])))

    out_pitch_png = out_png.split(".")[0] + "_pitch.png"
    out_energy_png = out_png.split(".")[0] + "_energy.png"

    ####### Pitch Contour ############
    compare_pitch_contour(
        pitch_dict_for_vis,
        rc_num,
        legend_mark=legend_mark,
        legend_color=legend_color,
        legend_linestyle=legend_linestyle,
        xy_label_sub=xy_label_sub,
        xtickslab=syn_phones,
        out_png=out_pitch_png,
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
        out_png=out_energy_png,
        title="Energy contour of speech synthesized on reference, emoMix, proposed ({})".format(title_extra),
        show_txt=show_ref_phone_on_line
    )


def vis_psd_enh(pitch_dict_for_vis, prosody_dict, out_png, txt_id, show_ref_phone_on_line=True, aux_range=None):
    rc_num = (3, 2)
    legend_mark = ("*", "v", ".")  # reference, guide, ropePhone_guide
    legend_color = ("grey", "lightblue", "blue")
    legend_linestyle = ("dashed", "solid", "solid")
    xy_label_sub = ("Frame", "Hz")

    #txt_n = len(prosody_dict["Angry"]["reference"]["phonemes"])   # should have ten
    real_text_id = max([i if txt_name[-1] == str(txt_id) else -1 for i, txt_name in
                        enumerate(prosody_dict["Angry"]["reference"]["speechid"])])  # psd["pitch"] is not sorted
    syn_phones = prosody_dict["Angry"]["reference"]["phonemes"][real_text_id]   # syn_phones should be same for all emotion

    title_extra = f"txt{txt_id}"
    print("3.1. do pitch comaration visualization on {}".format(" ".join(
        prosody_dict["Angry"]["reference"]["phonemes"][txt_id])))

    compare_pitch_contour(
        pitch_dict_for_vis,
        rc_num,
        legend_mark=legend_mark,
        legend_color=legend_color,
        legend_linestyle=legend_linestyle,
        xy_label_sub=xy_label_sub,
        xtickslab=syn_phones,
        out_png=out_png.format("enhance", title_extra.split(" ")[0]),
        title="Pitch contour of speech synthesized on reference, emoMix, proposed ({})".format(title_extra),
        show_txt=show_ref_phone_on_line,
        aux_range=aux_range,
    )

def vis_emo_crossAttn(attn_map_for_vis, out_png, show_txt=0, tick_gran="syllable"):
    """
    Show N pic for each model, each model include M subplot for each emotion
    Args:
        attn_map_for_vis:
        prosody_dict:
        out_png:
        show_txt:
        tick_gran:

    Returns:
    """
    # attn_map_for_vis -> {"model1": {"emo1": [np.array([syn_frames, ref_frames]), durations, phonemes] }}}
    for model_n, emo_attn_dict in attn_map_for_vis.items():  # {"model1": {"emo1": np.array([syn_frames, ref_frames])}}}
        rc_num = (3, 2)
        xy_label_sub = ("Reference frames", "Synthesis frames")
        title_extra = "non_parallel"
        show_attn_map(
            emo_attn_dict,
            rc_num,
            xlabel=xy_label_sub[0],
            ylabel=xy_label_sub[1],
            out_png=out_png.split(".")[0] + "_" + model_n + ".png",
            # "result/{}_similarity_{}.png"
            title="Attention map in crossAttention after masking ({})".format(title_extra),
            tick_gran=tick_gran
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
        show_txt=True,
        aux_range=None
):
  """
  args:
    sub_leg_data_dict:
    {emotion:{modelA: [1, 2, 5], modelB: [2, 3, 4]...}}
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
      
    if aux_range is not None:
        for ax_x in aux_range:
            axes[r, c].axvline(x=ax_x, color="blue", linestyle="--", linewidth=0.3)
    n += 1
  fig.delaxes(axes[2, 1])
  fig.suptitle(title)
  fig.savefig(out_png, dpi=300)

def show_attn_map(
        sub_data_dict,
        rc_num,
        out_png,
        title="attn_map",
        tick_gran="phoneme",
        xlabel="Reference frames",
        ylabel="Synthesis frames",
        rotation=45,
        xy_label_font_size=12,
        need_auxline=False,
        need_rectangle=True,
        stress_xyLable_index=2,   # set to 100 if not use
        rect_line_width=0.5,
        rect_line_style="--"
):
    """

    Args:
    sub_data_dict:  {"emo1": [attn, syn_phone_durs, syn_phones, ref_phone_durs, ref_phones] }}}
    """
    r_num, c_num = rc_num
    fig, axes = plt.subplots(r_num, c_num, figsize=(10 * (r_num / c_num), 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    fontsize = 12
    n = 0

    for emo, contents in sub_data_dict.items():
        r = int(n / c_num)
        c = int(n % c_num)
        axes[r, c].set_title(emo, fontsize=fontsize)
        attn, syn_phone_durs, syn_phones, ref_phone_durs, ref_phones = contents
        pc = axes[r, c].pcolor(attn, cmap=plt.cm.Blues, alpha=0.9)

        #### 1. Change granularity of x, y ticks, labels
        if tick_gran == "syllable":
            syn_labels, syn_durs = extend_phone2syl(syn_phones, syn_phone_durs) # change phone to syllable
            ref_labels, ref_durs = extend_phone2syl(ref_phones, ref_phone_durs)
        else:
            syn_labels, syn_durs = clean_phone(syn_phones, syn_phone_durs)  # clean "", 11
            ref_labels, ref_durs = clean_phone(ref_phones, ref_phone_durs)

        #### 2. Set xy ticks, ticklabels, and labels
        syn_durs_inc, ref_durs_inc = [0], [0]
        syn_durs_inc.extend([sum(syn_durs[:i + 1]) for i in range(len(syn_durs))])
        ref_durs_inc.extend([sum(ref_durs[:i + 1]) for i in range(len(ref_durs))])
        syn_labels.append("")  # to align label to the left
        ref_labels.append("")  # to align label to the left
        #print("syn_durs_inc, syn_labels: {} {}".format(syn_durs_inc, syn_labels))
        #print("ref_durs_inc, ref_labels: {} {}".format(ref_durs_inc, ref_labels))
        if len(syn_durs_inc) != len(syn_labels) or len(ref_durs_inc) != len(ref_labels):
            raise IOError("durs and phones should have same lens {} {} {} {}".format(
                len(syn_durs_inc), len(syn_labels), len(ref_durs_inc), len(ref_labels)))
        axes[r, c].set_xticks(syn_durs_inc)
        axes[r, c].set_xticklabels(labels=syn_labels, fontsize=xy_label_font_size, rotation=rotation, ha="left")
        axes[r, c].set_yticks(ref_durs_inc)
        axes[r, c].set_yticklabels(labels=ref_labels, fontsize=xy_label_font_size, rotation=rotation, va="center")
        axes[r, c].set_xlabel(xlabel)
        axes[r, c].set_ylabel(ylabel)
        # set bold xyLabels
        for i, xlab in enumerate(axes[r, c].get_xticklabels()):
            if i == stress_xyLable_index:
                xlab.set_fontweight("bold")
        for i, ylab in enumerate(axes[r, c].get_yticklabels()):
            if i == stress_xyLable_index:
                ylab.set_fontweight("bold")

        #### 3. set auxiliary lines and rectangle
        if need_auxline:
            for ax_x in syn_durs_inc:
                axes[r, c].axvline(x=ax_x, color="blue", linestyle="--", linewidth=0.3)
            for ax_y in ref_durs_inc:
                axes[r, c].axhline(y=ax_y, color="blue", linestyle="--", linewidth=0.3)
        if need_rectangle:
            xywh_list = convert_xydur_xybox(syn_durs_inc, ref_durs_inc)  # get rectangle coordinate
            for i, (x, y, w, h) in enumerate(xywh_list):
                if i == stress_xyLable_index:
                    rect_line_width = 1.0
                    rect_line_style = "-"
                axes[r, c].add_patch(plt.Rectangle((x, y), w, h, ls=rect_line_style, ec="red", fc="none", linewidth=rect_line_width))
        fig.colorbar(pc, ax=axes[r, c])
        n += 1

    fig.suptitle(title)
    fig.delaxes(axes[2, 1])
    fig.savefig(out_png, dpi=300)



def show_two_attn_map(
        sub_data_dict,
        rc_num,
        out_png,
        title="attn_map",
        tick_gran="phoneme",
        xlabel="Reference frames",
        ylabel="Synthesis frames",
        rotation=45,
        xy_label_font_size=12,
        need_auxline=False,
        need_rectangle=True,
        stress_xyLable_index=2,   # set to 100 if not use
        rect_line_width=0.5,
        rect_line_style="--"
):
    """

    Args:
    sub_data_dict:  {"emo1": [attn, syn_phone_durs, syn_phones, ref_phone_durs, ref_phones] }}}
    """
    r_num, c_num = rc_num
    fig, axes = plt.subplots(r_num, c_num, figsize=(30, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    fontsize = 12
    n = 0

    for emo, contents in sub_data_dict.items():
        r = int(n / c_num)
        c = int(n % c_num)
        axes[c].set_title(emo, fontsize=fontsize)
        attn, syn_phone_durs, syn_phones, ref_phone_durs, ref_phones = contents
        pc = axes[c].pcolor(attn, cmap=plt.cm.Blues, alpha=0.9)

        #### 1. Change granularity of x, y ticks, labels
        if tick_gran == "syllable":
            syn_labels, syn_durs = extend_phone2syl(syn_phones, syn_phone_durs) # change phone to syllable
            ref_labels, ref_durs = extend_phone2syl(ref_phones, ref_phone_durs)
        else:
            syn_labels, syn_durs = clean_phone(syn_phones, syn_phone_durs)  # clean "", 11
            ref_labels, ref_durs = clean_phone(ref_phones, ref_phone_durs)

        #### 2. Set xy ticks, ticklabels, and labels
        syn_durs_inc, ref_durs_inc = [0], [0]
        syn_durs_inc.extend([sum(syn_durs[:i + 1]) for i in range(len(syn_durs))])
        ref_durs_inc.extend([sum(ref_durs[:i + 1]) for i in range(len(ref_durs))])
        syn_labels.append("")  # to align label to the left
        ref_labels.append("")  # to align label to the left
        #print("syn_durs_inc, syn_labels: {} {}".format(syn_durs_inc, syn_labels))
        #print("ref_durs_inc, ref_labels: {} {}".format(ref_durs_inc, ref_labels))
        if len(syn_durs_inc) != len(syn_labels) or len(ref_durs_inc) != len(ref_labels):
            raise IOError("durs and phones should have same lens {} {} {} {}".format(
                len(syn_durs_inc), len(syn_labels), len(ref_durs_inc), len(ref_labels)))
        axes[c].set_xticks(syn_durs_inc)
        axes[c].set_xticklabels(labels=syn_labels, fontsize=xy_label_font_size, rotation=rotation, ha="left")
        axes[c].set_yticks(ref_durs_inc)
        axes[c].set_yticklabels(labels=ref_labels, fontsize=xy_label_font_size, rotation=rotation, va="center")
        axes[c].set_xlabel(xlabel)
        axes[c].set_ylabel(ylabel)
        # set bold xyLabels
        for i, xlab in enumerate(axes[c].get_xticklabels()):
            if i == stress_xyLable_index:
                xlab.set_fontweight("bold")
        for i, ylab in enumerate(axes[c].get_yticklabels()):
            if i == stress_xyLable_index:
                ylab.set_fontweight("bold")

        #### 3. set auxiliary lines and rectangle
        if need_auxline:
            for ax_x in syn_durs_inc:
                axes[c].axvline(x=ax_x, color="blue", linestyle="--", linewidth=0.3)
            for ax_y in ref_durs_inc:
                axes[c].axhline(y=ax_y, color="blue", linestyle="--", linewidth=0.3)
        if need_rectangle:
            xywh_list = convert_xydur_xybox(syn_durs_inc, ref_durs_inc)  # get rectangle coordinate
            for i, (x, y, w, h) in enumerate(xywh_list):
                if i == stress_xyLable_index:
                    rect_line_width = 1.0
                    rect_line_style = "-"
                axes[c].add_patch(plt.Rectangle((x, y), w, h, ls=rect_line_style, ec="red", fc="none", linewidth=rect_line_width))
        fig.colorbar(pc, ax=axes[c])
        n += 1
    fig.suptitle(title)
    fig.savefig(out_png, dpi=300)
