from pymcd.mcd import Calculate_MCD
import numpy as np
import os
mcd_toolbox = Calculate_MCD(MCD_mode="dtw")


def interpolate_nan(array_like):
    array_like = np.array(array_like)
    array = array_like.copy()
    nans = np.isnan(array)
    def get_x(a):
        return a.nonzero()[0]
    array[nans] = np.interp(get_x(nans), get_x(~nans), array[~nans])
    return array

def statcz_psd_mcd(prosody_dict, base_dir):
    """
    For parallel test
    Args:
        prosody_dict:  {"spk": {"emo1": {"model1": {"psd/phone/speechid": [[], []]}}}}

    Returns:
        psd_mcd_stat_res: {"spk": "emo1": {"model1": [[p_diff], [e_diff], [mcd_res]]}}}}
    """
    psd_mcd_stat_res = {}

    #models_name = [model_n for model_n in prosody_dict["Angry"].keys().tolist() if model_n != "reference"]
    diff_func = lambda x, y: np.mean(np.abs(np.array(x) - np.array(y)))
    mean_func = lambda x: np.mean(np.array(x))

    for spk, emo_model_psd in prosody_dict.items():
        if spk not in psd_mcd_stat_res.keys():
            psd_mcd_stat_res[spk] = dict()
        for emo, model_psd in emo_model_psd.items():
            if emo not in psd_mcd_stat_res[spk].keys():
                psd_mcd_stat_res[spk][emo] = dict()
            for model_n, psd_phone_sid in model_psd.items():
                if model_n != "reference":
                    if model_n not in psd_mcd_stat_res[spk][emo].keys():
                        psd_mcd_stat_res[spk][emo][model_n] = []
                    #psd_len = len(psd)
                    #assert psd_len == len(prosody_dict[emo]["reference"])
                    # statcz psd
                    wav_num = len(psd_phone_sid["pitch"])

                    #c1 = psd_phone_sid["pitch"][0]
                    #c2 = prosody_dict[emo]["reference"]["pitch"][0]

                    ### check
                    """
                    for i in range(wav_num):
                        if not np.all(np.isnan(psd_phone_sid["pitch"][i]) == False):
                            print(psd_phone_sid["pitch"][i])
                    """

                    p_diffs = []
                    e_diffs = []
                    for i in range(wav_num):
                        syn_pitch_contour = interpolate_nan(psd_phone_sid["pitch"][i])
                        ref_pitch_contour = interpolate_nan(prosody_dict[spk][emo]["reference"]["pitch"][i])
                        min_len = min(len(syn_pitch_contour), len(ref_pitch_contour))
                        p_diffs.append(diff_func(syn_pitch_contour[:min_len], ref_pitch_contour[:min_len]))

                        syn_energy_contour = interpolate_nan(psd_phone_sid["energy"][i])
                        ref_energy_contour = interpolate_nan(prosody_dict[spk][emo]["reference"]["energy"][i])
                        min_len = min(len(syn_energy_contour), len(ref_energy_contour))
                        e_diffs.append(diff_func(syn_energy_contour[:min_len], ref_energy_contour[:min_len]))

                    #p_diffs = [diff_func(interpolate_nan(psd_phone_sid["pitch"][i]), interpolate_nan(prosody_dict[spk][emo]["reference"]["pitch"][i]))
                    #          for i in range(wav_num)]
                    #e_diffs = [diff_func(interpolate_nan(psd_phone_sid["energy"][i]), interpolate_nan(prosody_dict[spk][emo]["reference"]["energy"][i]))
                    #          for i in range(wav_num)]

                    p_diffs_mean = mean_func(p_diffs)
                    e_diffs_mean = mean_func(e_diffs)
                    psd_mcd_stat_res[spk][emo][model_n].append(p_diffs_mean)
                    psd_mcd_stat_res[spk][emo][model_n].append(e_diffs_mean)

                    # statcz mcd
                    wav_dir = os.path.join(base_dir, model_n)
                    ref_dir = os.path.join(base_dir, "reference")
                    mcds = [
                        mcd_toolbox.calculate_mcd(wav_dir + "/" + psd_phone_sid["speechid"][i] + ".wav",
                                                  ref_dir + "/" + prosody_dict[spk][emo]["reference"]["speechid"][i]  + ".wav")
                        for i in range(wav_num)]
                    mcd_res_mean = mean_func(mcds)
                    psd_mcd_stat_res[spk][emo][model_n].append(mcd_res_mean)
    return psd_mcd_stat_res



