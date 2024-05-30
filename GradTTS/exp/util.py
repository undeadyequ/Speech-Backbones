from SER import predict_emotion
import librosa
from pymcd.mcd import Calculate_MCD


def get_pitch_match_score(pitch1, pitch2):
    pitch_score = 0
    return pitch_score


if __name__ == '__main__':
    # test MCD

    # instance of MCD class
    # three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")

    # two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively
    mcd_value = mcd_toolbox.calculate_mcd("../exp/sample_21st.wav", "../exp/sample_22nd.wav")
    print(mcd_value)

