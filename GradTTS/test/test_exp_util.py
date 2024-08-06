from GradTTS.exp.util import get_ref_pRange


def test_get_ref_pRange():
    ref_id = "0019_000403"
    ref_nRange = get_ref_pRange(2, ref_id)
    print(ref_nRange)