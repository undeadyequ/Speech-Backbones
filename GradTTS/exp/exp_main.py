from GradTTS.model.util_maskCreation import create_p2p_mask
from GradTTS.inference_cond import inference_p2p_transfer_by_attnMask
from GradTTS.exp.utils import compute_phoneme_mcd


def evaluate_mcd_main(
        referStyle_phone_oneText_dict,
        eval_syntext_by_p,
        tgt_pStart,
        tgt_pEnd,
        ref1_pStart,
        ref1_pEnd,
        ref2_pStart,
        ref2_pEnd
):
    """
    referStyle_phone_oneText_dict:
    eval_syntext_by_p: evaluation syntext divided by phomeme
    """
    # read eval_syntext_by_p from file ...
    ref1_mel, ref1_emo, ref1_spk = "", "", ""
    for pStyle, pDict in referStyle_phone_oneText_dict.items():
        # test each phoneme style
        for p, pTuple in pDict.items:
            # test each phoneme
            ref2_mel, ref2_emo, ref2_spk, ref2_start, ref2_end = pTuple
            for syn_text_tuple in eval_syntext_by_p[p]:
                syn_text, pStart, pEnd = syn_text_tuple

                ref1_mask = create_p2p_mask(
                    tgt_size=syn_text.shape[1],
                    ref_size=ref1_mel.shape[1],
                    tgt_range=(tgt_pStart, tgt_pEnd),
                    ref_range=(ref1_pStart, ref1_pEnd)
                )

                ref2_mask = create_p2p_mask(
                    tgt_size=syn_text.shape[1],
                    ref_size=ref2_mel.shape[1],
                    tgt_range=(pStart, pEnd),
                    ref_range=(ref2_start, ref2_end)
                )

                syn_speech, attn_hardMask = inference_p2p_transfer_by_attnMask(
                    syn_text, pStart, pEnd,
                    ref1_mel, ref1_emo, ref1_spk, ref1_pStart, ref1_pEnd, ref1_mask,
                    ref2_mel, ref2_emo, ref2_spk, ref2_pStart, ref2_pEnd, ref2_mask,
                    p2p_mode="crossAttn"
                )

                syn_pStart, syn_pEnd = get_frameLevel_pstart_end(attn_hardMask)
                syn_speech_phoneme_mels = read_mels(syn_speech)[syn_pStart: syn_pEnd]

                ref2_speech_phoneme_mels = ref2_mel[ref2_start: ref2_end]

                phonem_mcd = compute_phoneme_mcd(
                    syn_speech_phoneme_mels,
                    ref2_speech_phoneme_mels
                )


def get_frameLevel_pstart_end(attn_hardMask):
    """

    Args:
        attn_hardMask: (1,1, l_q * 80, l_k). l_q and l_k is frame-level target and reference speech.
        Mask value equals True means it is marked
    Returns:
        syn_pStart: Masked phoneme start at frame-level
        syn_pEnd: Masked phoneme start at frame-level

    """
    pass

def eval_psd_main():
    pass
