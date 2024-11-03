import json
import os



# transfer prosody style of phoneme to target phoneme, deviding by style and phoneme sound
ref1_mel, emo, spk, ref1_pStart, ref1_pEnd = "", "", "", "", ""

origin_syn_text = ["syn_text1", ...]
eval_p = ["ar", ...]
eval_syntext_by_p = {
  "ar": [
      ("syn_text1", "p_start1", "p_end1"),
      ("syn_text2", "p_start2", "p_end2"),
      ...
      ],
  "ir": [],
}

referStyle_phone_dict = {
  "high_energy":
  {
    "ar": [
      (id, emo, spk, "mels", "syn_text1", "start", "end", "sign_score"),
      ...,
    ],
    "ir": [
      ...
    ]
  },
  "low_energy": {},
  "increase_energy": {},
  "decrease_energy": {},
  "aboveTurn_energy": {},
  "belowTurn_energy": {},
  "..._pitch": {},
}


########## Utterance-level test #########



###########  phoneme-level test #########
def classify_synText_by_phoneme(
  origin_syn_text,
  evalute_p
  ):
  eval_syntext_by_p = {}
  # extract phoneme and start, end (the start, end extraction is rule-based)
  return eval_syntext_by_p

def select_syn_text_by_p(
  origin_syn_text_f,
  eval_type,
):
  with open(origin_syn_text_f, "r") as f:
    f.writelines(origin_syn_text)
  if eval_type == "mcd":
    eval_syntext_by_p = classify_synText_by_phoneme(origin_syn_text, eval_p)
    # get statisitics of eval_syntext_by_p
    [print("{} included in {} syn_text".format(k, len(v))) for k, v in eval_syntext_by_p.items()]
  else:
    print("on buiding")
  # write to json
  eval_syntext_by_p_f = os.path.base(origin_syn_text_f).split(".")[0] + "_by_p.json"
  with open(eval_syntext_by_p_f, "r") as f:
    json.dumps(eval_syntext_by_p)
