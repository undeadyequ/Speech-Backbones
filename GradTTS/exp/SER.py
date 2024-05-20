import torch
from transformers import AutoProcessor, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import numpy as np
from pydub import AudioSegment
from torch import nn

# https://github.com/ehcalabres/EMOVoice
# the preprocessor was derived from https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english
# processor1 = AutoProcessor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
# ^^^ no preload model available for this model (above), but the `feature_extractor` works in place
model1 = Wav2Vec2ForSequenceClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
id2label = {
    "0": "angry",
    "1": "calm",
    "2": "disgust",
    "3": "fearful",
    "4": "happy",
    "5": "neutral",
    "6": "sad",
    "7": "surprised"
}


def predict_emotion(audio_file,
                    sr=16000):
    sound = AudioSegment.from_file(audio_file)
    sound = sound.set_frame_rate(sr)
    sound_array = np.array(sound.get_array_of_samples())
    # this model is VERY SLOW, so best to pass in small sections that contain
    # emotional words from the transcript. like 10s or less.
    # how to make sub-chunk  -- this was necessary even with very short audio files
    # test = torch.tensor(input.input_values.float()[:, :100000])

    input = feature_extractor(
        raw_speech=sound_array,
        sampling_rate=sr,
        padding=True,
        return_tensors="pt")

    result = model1.forward(input.input_values.float())
    # making sense of the result

    result_pred = nn.functional.softmax(result.logits, dim=-1)
    interp = dict(zip(id2label.values(), list(round(float(i), 4) for i in result_pred[0])))

    # get res emotion
    res_emo_id = int(torch.argmax(result_pred, dim=-1)[0])
    res_emo = id2label[str(res_emo_id)]
    return interp, res_emo


if __name__ == '__main__':
    audio_f = "/home/rosen/data/ESD/0001/Angry/test/0001_000371.wav"
    audio_f = "/home/rosen/Project/Speech-Backbones/GradTTS/logs/gradtts_crossSelf_puncond_n1_neworder_fixmask/interpTemp/Angry_Happy/sample_2_1st.wav"
    interp, res_emo = predict_emotion(audio_f, sr=22050)
    print(res_emo)

    """
    AutoModelForAudioClassification:
    "{'angry': 0.028, 'calm': -0.0703, 'disgust': -0.0061, 'fearful': -0.06, 'happy': -0.0262, 'neutral': -0.0657, 'sad': -0.0805, 'surprised': -0.011}"
    Wav2Vec2ForSequenceClassification:
    {'angry': -0.0068, 'calm': 0.0255, 'disgust': 0.0612, 'fearful': 0.0293, 'happy': -0.1165, 'neutral': 0.0442, 'sad': 0.0496, 'surprised': 0.0756}

    """