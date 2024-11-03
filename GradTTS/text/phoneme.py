from GradTTS.text import text_to_sequence, cmudict, text_to_arpabet



cmu = cmudict.CMUDict('/home/rosen/Project/Speech-Backbones/GradTTS/resources/cmu_dictionary')
def get_phoneme(text: str) -> str:
    """
    Get phoneme from text
    """
    ref2_phnms = text_to_arpabet(text, cleaner_names=["english_cleaners"], dictionary=cmu)  # ["p1", "p2"]
    return ref2_phnms

def get_phonemeID(text: str, phoneme: str) -> int:
    """
    get phonemeID from text
    """
    phonemeID = 1
    return phonemeID

