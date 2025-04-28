""" from https://github.com/keithito/tacotron """
import re
from GradTTS.text import cleaners, cmudict
from GradTTS.text.symbols import symbols
from GradTTS.text.syllable import extend_phone2syl, search_syllable_index
from GradTTS.utils import intersperse

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

cmu = cmudict.CMUDict('/home/rosen/Project/Speech-Backbones/GradTTS/resources/cmu_dictionary')

text2id_iterp = lambda x: intersperse(text_to_sequence(x, dictionary=cmu), len(symbols)) # for training
text2phone_iterp = lambda x: intersperse(text_to_arpabet(x, dictionary=cmu), "")         # for visualization
text2sylstart_interp = lambda x: search_syllable_index(text2phone_iterp(x))         # for training  and vis


def get_arpabet(word, dictionary):
    word_arpabet = dictionary.lookup(word)
    if word_arpabet is not None:
        return "{" + word_arpabet[0] + "}"
    else:
        return word

def text_to_arpabet(text, cleaner_names=["english_cleaners"], dictionary=None):
    sequence = []
    space = _symbols_to_sequence(' ')
    if dictionary is not None:
        clean_text = _clean_text(text, cleaner_names)
        clean_text = [get_arpabet(w, dictionary) for w in clean_text.split(" ")]  # [{AH I}, forest!]
        for i in range(len(clean_text)):
            t = clean_text[i]
            if t.startswith("{"):
                sequence += t[1:-1].split(" ")   # [AH, I, forest!]
            else:
                sequence += [s for s in t if _should_keep_symbol(s)]
            sequence += space   # # [AH, "", I, "", forest!]
        sequence = sequence[:-1] if sequence[-1] == space[0] else sequence
    return sequence

def phoneme_to_sequence(phonemes):
    """Converts phonemes to a sequence of IDs corresponding to the symbols in the text.
    Args:
        phoneme: HH ER0 K AY1 N D AH0 N D F ER1 M G L AE1 N S
    Returns:
    """
    p_ids = []
    phonemes_l = phonemes.split()
    for p in phonemes_l:
        p_ids += _arpabet_to_sequence(p)
    return p_ids

def text_to_sequence(text, cleaner_names=["english_cleaners"], dictionary=None):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    Example: id of [AH, "", I, "", f, o, r, e, s, t, !]
    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      dictionary: arpabet class with arpabet dictionary

    Returns:
      List of integers corresponding to the symbols in the text
    '''
    sequence = []
    space = _symbols_to_sequence(' ')
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            clean_text = _clean_text(text, cleaner_names)
            if dictionary is not None:
                clean_text = [get_arpabet(w, dictionary) for w in clean_text.split(" ")]
                #print("phomeme: {}".format(clean_text))
                for i in range(len(clean_text)):
                    t = clean_text[i]
                    if t.startswith("{"):
                        sequence += _arpabet_to_sequence(t[1:-1])
                    else:
                        sequence += _symbols_to_sequence(t)
                    sequence += space
            else:
                sequence += _symbols_to_sequence(clean_text)
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)
  
    # remove trailing space
    if dictionary is not None:
        sequence = sequence[:-1] if sequence[-1] == space[0] else sequence
    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != '_' and s != '~'
