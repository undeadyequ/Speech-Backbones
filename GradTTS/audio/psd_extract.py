import opensmile
from pathlib import Path
import os
import numpy as np

opensmile_features = opensmile.FeatureSet.GeMAPSv01a
opensmile_functional = opensmile.FeatureLevel.Functionals

def extract_opensmile(wav_dir, out_dir):
    print("smile extraction started!")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    smile = opensmile.Smile(
        feature_set=opensmile_features,
        feature_level=opensmile_functional,
    )
    dir_path = Path(wav_dir)
    for audio_f in dir_path.rglob("*.wav"):
        psd_reps = smile.process_file(audio_f)
        psd_reps = psd_reps.squeeze(0)
        audio_name = audio_f.stem
        np.save(os.path.join(out_dir, audio_name + ".npy"), psd_reps)
    print("smile extraction finished!")
