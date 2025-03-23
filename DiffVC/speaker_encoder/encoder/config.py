""" from https://github.com/CorentinJ/Real-Time-Voice-Cloning """

librispeech_datasets = {
    "train": {
        "clean": ["LibriSpeech/train-clean-100", "LibriSpeech/train-clean-360"],
        "bk": ["LibriSpeech/train-bk-500"]
    },
    "test": {
        "clean": ["LibriSpeech/test-clean"],
        "bk": ["LibriSpeech/test-bk"]
    },
    "dev": {
        "clean": ["LibriSpeech/dev-clean"],
        "bk": ["LibriSpeech/dev-bk"]
    },
}
libritts_datasets = {
    "train": {
        "clean": ["LibriTTS/train-clean-100", "LibriTTS/train-clean-360"],
        "bk": ["LibriTTS/train-bk-500"]
    },
    "test": {
        "clean": ["LibriTTS/test-clean"],
        "bk": ["LibriTTS/test-bk"]
    },
    "dev": {
        "clean": ["LibriTTS/dev-clean"],
        "bk": ["LibriTTS/dev-bk"]
    },
}
voxceleb_datasets = {
    "voxceleb1" : {
        "train": ["VoxCeleb1/wav"],
        "test": ["VoxCeleb1/test_wav"]
    },
    "voxceleb2" : {
        "train": ["VoxCeleb2/dev/aac"],
        "test": ["VoxCeleb2/test_wav"]
    }
}

other_datasets = [
    "LJSpeech-1.1",
    "VCTK-Corpus/wav48",
]

anglophone_nationalites = ["australia", "canada", "ireland", "uk", "usa"]
