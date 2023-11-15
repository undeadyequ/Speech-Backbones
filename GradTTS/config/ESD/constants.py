dimension_name = [
        "F0semitoneFrom27.5Hz_sma3nz_amean",
        "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
        "F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
        "F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
        "F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
        "F0semitoneFrom27.5Hz_sma3nz_pctlrange0 - 2",
        "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
        "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
        "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",
        "F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope",
        "loudness_sma3_amean",
        "loudness_sma3_stddevNorm",
        "loudness_sma3_percentile20.0",
        "loudness_sma3_percentile50.0",
        "loudness_sma3_percentile80.0",
        "loudness_sma3_pctlrange0 - 2",
        "loudness_sma3_meanRisingSlope",
        "loudness_sma3_stddevRisingSlope",
        "loudness_sma3_meanFallingSlope",
        "loudness_sma3_stddevFallingSlope",
        "jitterLocal_sma3nz_amean",
        "jitterLocal_sma3nz_stddevNorm",
        "shimmerLocaldB_sma3nz_amean",
        "shimmerLocaldB_sma3nz_stddevNorm",
        "HNRdBACF_sma3nz_amean",
        "HNRdBACF_sma3nz_stddevNorm",
        "logRelF0 - H1 - H2_sma3nz_amean",
        "logRelF0 - H1 - H2_sma3nz_stddevNorm",
        "logRelF0 - H1 - A3_sma3nz_amean",
        "logRelF0 - H1 - A3_sma3nz_stddevNorm",
        "F1frequency_sma3nz_amean",
        "F1frequency_sma3nz_stddevNorm",
        "F1bandwidth_sma3nz_amean",
        "F1bandwidth_sma3nz_stddevNorm",
        "F1amplitudeLogRelF0_sma3nz_amean",
        "F1amplitudeLogRelF0_sma3nz_stddevNorm",
        "F2frequency_sma3nz_amean",
        "F2frequency_sma3nz_stddevNorm",
        "F2amplitudeLogRelF0_sma3nz_amean",
        "F2amplitudeLogRelF0_sma3nz_stddevNorm",
        "F3frequency_sma3nz_amean",
        "F3frequency_sma3nz_stddevNorm",
        "F3amplitudeLogRelF0_sma3nz_amean",
        "F3amplitudeLogRelF0_sma3nz_stddevNorm",
        "alphaRatioV_sma3nz_amean",
        "alphaRatioV_sma3nz_stddevNorm",
        "hammarbergIndexV_sma3nz_amean",
        "hammarbergIndexV_sma3nz_stddevNorm",
        "slopeV0 - 500_sma3nz_amean",
        "slopeV0 - 500_sma3nz_stddevNorm",
        "slopeV500 - 1500_sma3nz_amean",
        "slopeV500 - 1500_sma3nz_stddevNorm",
        "alphaRatioUV_sma3nz_amean",
        "hammarbergIndexUV_sma3nz_amean",
        "slopeUV0 - 500_sma3nz_amean",
        "slopeUV500 - 1500_sma3nz_amean",
        "loudnessPeaksPerSec",
        "VoicedSegmentsPerSec",
        "MeanVoicedSegmentLengthSec",
        "StddevVoicedSegmentLengthSec",
        "MeanUnvoicedSegmentLength",
        "StddevUnvoicedSegmentLength",
    ]

# this is set by get_optimal_grpNum
emo_optimal_clusterN_ver1 = {
        "Angry": 3,
        "Surprise": 2,
        "Sad": 4,
        "Neutral": 2,
        "Happy": 2
}

emo_optimal_clusterN = {
        "Angry": 2,
        "Surprise": 2,
        "Sad": 3,
        "Neutral": 2,
        "Happy": 2
}

emo_optimal_clusterN_test = {
        "Angry": 3,
        "Surprise": 3,
        "Sad": 4,
        "Neutral": 3,
        "Happy": 3
}

# The number of contribute dims
emo_contri_dimN = {
        "Angry": 2,
        "Surprise": 1,
        "Sad": 2,
        "Neutral": 1,
        "Happy": 2
}

emo_num = {
    "Angry": 0,
    "Surprise": 1,
    "Sad": 2,
    "Neutral": 3,
    "Happy": 4
}

num_emo = {
    0: "Angry",
    1: "Surprise",
    2: "Sad",
    3: "Neutral",
    4: "Happy"
}

emo_contrdim = {
        "Neutral": (6, 7),
        "Sad": (6, 7),
        "Angry": (6, 51),
        "Surprise": (6, 7),
        "Happy": (6, 7),
}