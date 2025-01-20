import re
import numpy as np


MIDI_BASE_CLASS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
# _MORPHETIC_BASE_CLASS = {'c': 0, 'd': 1, 'e': 2, 'f': 3, 'g': 4, 'a': 5, 'b': 6}
# _MORPHETIC_OCTAVE = {0: 32, 1: 39, 2: 46, 3: 53, 4: 60, 5: 67, 6: 74, 7: 81, 8: 89}
ALTER_SIGNS = {None: "", 0: "", 1: "#", 2: "x", -1: "b", -2: "bb"}

DUMMY_PS_BASE_CLASS = {
    0: ("c", 0),
    1: ("c", 1),
    2: ("d", 0),
    3: ("d", 1),
    4: ("e", 0),
    5: ("f", 0),
    6: ("f", 1),
    7: ("g", 0),
    8: ("g", 1),
    9: ("a", 0),
    10: ("a", 1),
    11: ("b", 0),
}

MEI_DURS_TO_SYMBOLIC = {
    "long": "long",
    "0": "breve",
    "breve": "breve",
    "1": "whole",
    "2": "half",
    "4": "quarter",
    "8": "eighth",
    "16": "16th",
    "32": "32nd",
    "64": "64th",
    "128": "128th",
    "256": "256th",
}

SYMBOLIC_TO_INT_DURS = {
    "long": 0.25,
    "breve": 0.5,
    "whole": 1,
    "half": 2,
    "quarter": 4,
    "eighth": 8,
    "16th": 16,
    "32nd": 32,
    "64th": 64,
    "128th": 128,
    "256th": 256,
}

LABEL_DURS = {
    "long": 16,
    "breve": 8,
    "whole": 4,
    "half": 2,
    "h": 2,
    "quarter": 1,
    "q": 1,
    "eighth": 1 / 2,
    "e": 1 / 2,
    "16th": 1 / 4,
    "32nd": 1 / 8.0,
    "64th": 1 / 16,
    "128th": 1 / 32,
    "256th": 1 / 64,
}
DOT_MULTIPLIERS = (1, 1 + 1 / 2, 1 + 3 / 4, 1 + 7 / 8)
# DURS and SYM_DURS encode the same information as _LABEL_DURS and
# _DOT_MULTIPLIERS, but they allow for faster estimation of symbolic duration
# (estimate_symbolic duration). At some point we will probably do away with
# _LABEL_DURS and _DOT_MULTIPLIERS.
DURS = np.array(
    [
        1.5625000e-02,
        2.3437500e-02,
        2.7343750e-02,
        2.9296875e-02,
        3.1250000e-02,
        4.6875000e-02,
        5.4687500e-02,
        5.8593750e-02,
        6.2500000e-02,
        9.3750000e-02,
        1.0937500e-01,
        1.1718750e-01,
        1.2500000e-01,
        1.8750000e-01,
        2.1875000e-01,
        2.3437500e-01,
        2.5000000e-01,
        3.7500000e-01,
        4.3750000e-01,
        4.6875000e-01,
        5.0000000e-01,
        5.0000000e-01,
        7.5000000e-01,
        7.5000000e-01,
        8.7500000e-01,
        8.7500000e-01,
        9.3750000e-01,
        9.3750000e-01,
        1.0000000e00,
        1.0000000e00,
        1.5000000e00,
        1.5000000e00,
        1.7500000e00,
        1.7500000e00,
        1.8750000e00,
        1.8750000e00,
        2.0000000e00,
        2.0000000e00,
        3.0000000e00,
        3.0000000e00,
        3.5000000e00,
        3.5000000e00,
        3.7500000e00,
        3.7500000e00,
        4.0000000e00,
        6.0000000e00,
        7.0000000e00,
        7.5000000e00,
        8.0000000e00,
        1.2000000e01,
        1.4000000e01,
        1.5000000e01,
        1.6000000e01,
        2.4000000e01,
        2.8000000e01,
        3.0000000e01,
    ]
)

SYM_DURS = [
    {"type": "256th", "dots": 0},
    {"type": "256th", "dots": 1},
    {"type": "256th", "dots": 2},
    {"type": "256th", "dots": 3},
    {"type": "128th", "dots": 0},
    {"type": "128th", "dots": 1},
    {"type": "128th", "dots": 2},
    {"type": "128th", "dots": 3},
    {"type": "64th", "dots": 0},
    {"type": "64th", "dots": 1},
    {"type": "64th", "dots": 2},
    {"type": "64th", "dots": 3},
    {"type": "32nd", "dots": 0},
    {"type": "32nd", "dots": 1},
    {"type": "32nd", "dots": 2},
    {"type": "32nd", "dots": 3},
    {"type": "16th", "dots": 0},
    {"type": "16th", "dots": 1},
    {"type": "16th", "dots": 2},
    {"type": "16th", "dots": 3},
    {"type": "eighth", "dots": 0},
    {"type": "e", "dots": 0},
    {"type": "eighth", "dots": 1},
    {"type": "e", "dots": 1},
    {"type": "eighth", "dots": 2},
    {"type": "e", "dots": 2},
    {"type": "eighth", "dots": 3},
    {"type": "e", "dots": 3},
    {"type": "quarter", "dots": 0},
    {"type": "q", "dots": 0},
    {"type": "quarter", "dots": 1},
    {"type": "q", "dots": 1},
    {"type": "quarter", "dots": 2},
    {"type": "q", "dots": 2},
    {"type": "quarter", "dots": 3},
    {"type": "q", "dots": 3},
    {"type": "half", "dots": 0},
    {"type": "h", "dots": 0},
    {"type": "half", "dots": 1},
    {"type": "h", "dots": 1},
    {"type": "half", "dots": 2},
    {"type": "h", "dots": 2},
    {"type": "half", "dots": 3},
    {"type": "h", "dots": 3},
    {"type": "whole", "dots": 0},
    {"type": "whole", "dots": 1},
    {"type": "whole", "dots": 2},
    {"type": "whole", "dots": 3},
    {"type": "breve", "dots": 0},
    {"type": "breve", "dots": 1},
    {"type": "breve", "dots": 2},
    {"type": "breve", "dots": 3},
    {"type": "long", "dots": 0},
    {"type": "long", "dots": 1},
    {"type": "long", "dots": 2},
    {"type": "long", "dots": 3},
]

# Straight durs do not include copies for naming or dots, when searching they work better for base triplet types in `estimate_symbolic_duration`.
STRAIGHT_DURS = np.array(
    [
        4 / 256,
        4 / 128,
        4 / 64,
        4 / 32,
        4 / 16,
        4 / 8,
        4 / 4,
        4 / 2,
        4 / 1,
        4 / 0.5,
        4 / 0.25,
    ]
)

SYM_STRAIGHT_DURS = [
    {"type": "256th", "dots": 0},
    {"type": "128th", "dots": 0},
    {"type": "64th", "dots": 0},
    {"type": "32nd", "dots": 0},
    {"type": "16th", "dots": 0},
    {"type": "eighth", "dots": 0},
    {"type": "quarter", "dots": 0},
    {"type": "half", "dots": 0},
    {"type": "whole", "dots": 0},
    {"type": "breve", "dots": 0},
    {"type": "long", "dots": 0},
]

MAJOR_KEYS = [
    "Cb",
    "Gb",
    "Db",
    "Ab",
    "Eb",
    "Bb",
    "F",
    "C",
    "G",
    "D",
    "A",
    "E",
    "B",
    "F#",
    "C#",
]
MINOR_KEYS = [
    "Ab",
    "Eb",
    "Bb",
    "F",
    "C",
    "G",
    "D",
    "A",
    "E",
    "B",
    "F#",
    "C#",
    "G#",
    "D#",
    "A#",
]

TIME_UNITS = ["beat", "quarter", "sec", "div"]

NOTE_NAME_PATT = re.compile(r"([A-G]{1})([xb\#]*)(\d+)")

INTERVALCLASSES = [
    f"{specific}{generic}"
    for generic in [2, 3, 6, 7]
    for specific in ["dd", "d", "m", "M", "A", "AA"]
] + [
    f"{specific}{generic}"
    for generic in [1, 4, 5]
    for specific in ["dd", "d", "P", "A", "AA"]
]

INTERVAL_TO_SEMITONES = dict(
    zip(
        INTERVALCLASSES,
        [
            generic + specific
            for generic in [1, 3, 8, 10]
            for specific in [-2, -1, 0, 1, 2, 3]
        ]
        + [
            generic + specific
            for generic in [0, 5, 7]
            for specific in [-2, -1, 0, 1, 2]
        ],
    )
)


STEPS = {
    "C": 0,
    "D": 1,
    "E": 2,
    "F": 3,
    "G": 4,
    "A": 5,
    "B": 6,
    0: "C",
    1: "D",
    2: "E",
    3: "F",
    4: "G",
    5: "A",
    6: "B",
}


MUSICAL_BEATS = {6: 2, 9: 3, 12: 4}

# Standard tuning frequency of A4 in Hz
A4 = 440.0

COMPOSITE_DURS = np.array(
    [
        1 / 4 + 1 / 6,
        1 / 2 + 1 / 12,
        1 / 2 + 1 / 3,
        1 / 2 + 1 / 4 + 1 / 6,
        1 + 1 / 12,
        1 + 1 / 8,
        1 + 1 / 6,
        1 + 1 / 4,
        1 + 1 / 4 + 1 / 6,
        1 + 1 / 2 + 1 / 12,
        1 + 1 / 2 + 1 / 6,
        1 + 1 / 2 + 1 / 3,
        1 + 1 / 2 + 1 / 4 + 1 / 6,
        2 + 1 / 12,
        2 + 1 / 8,
        2 + 1 / 6,
        2 + 1 / 4,
        2 + 1 / 3,
        2 + 1 / 4 + 1 / 6,
        2 + 1 / 2,
        2 + 1 / 2 + 1 / 12,
        2 + 2 / 3,
        2 + 1 / 2 + 1 / 4,
        2 + 1 / 2 + 1 / 3,
        2 + 1 / 2 + 1 / 4 + 1 / 6,
        3 + 1 / 12,
        3 + 1 / 8,
        3 + 1 / 6,
        3 + 1 / 4,
        3 + 1 / 3,
        3 + 1 / 4 + 1 / 6,
        3 + 1 / 2 + 1 / 12,
        3 + 2 / 3,
        3 + 1 / 2 + 1 / 3,
        3 + 1 / 2 + 1 / 4 + 1 / 6,
    ]
)

SYM_COMPOSITE_DURS = [
    (
        {"type": "16th", "dots": 0},
        {"type": "16th", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "eighth", "dots": 0},
        {"type": "32nd", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "eighth", "dots": 0},
        {"type": "eighth", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "eighth", "dots": 1},
        {"type": "16th", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "quarter", "dots": 0},
        {"type": "32nd", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    ({"type": "quarter", "dots": 0}, {"type": "32nd", "dots": 0}),
    (
        {"type": "quarter", "dots": 0},
        {"type": "16th", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    ({"type": "quarter", "dots": 0}, {"type": "16th", "dots": 0}),
    (
        {"type": "quarter", "dots": 0},
        {"type": "16th", "dots": 0},
        {"type": "16th", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "quarter", "dots": 1},
        {"type": "32nd", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "quarter", "dots": 1},
        {"type": "16th", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "quarter", "dots": 1},
        {"type": "eighth", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "quarter", "dots": 2},
        {"type": "16th", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "half", "dots": 0},
        {"type": "32nd", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    ({"type": "half", "dots": 0}, {"type": "32nd", "dots": 0}),
    (
        {"type": "half", "dots": 0},
        {"type": "16th", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    ({"type": "half", "dots": 0}, {"type": "16th", "dots": 0}),
    (
        {"type": "half", "dots": 0},
        {"type": "eighth", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "half", "dots": 0},
        {"type": "16th", "dots": 0},
        {"type": "16th", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    ({"type": "half", "dots": 0}, {"type": "eighth", "dots": 0}),
    (
        {"type": "half", "dots": 0},
        {"type": "eighth", "dots": 0},
        {"type": "32nd", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "half", "dots": 0},
        {"type": "quarter", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    ({"type": "half", "dots": 0}, {"type": "eighth", "dots": 1}),
    (
        {"type": "half", "dots": 0},
        {"type": "eighth", "dots": 0},
        {"type": "eighth", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "half", "dots": 0},
        {"type": "eighth", "dots": 1},
        {"type": "16th", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "half", "dots": 1},
        {"type": "32nd", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    ({"type": "half", "dots": 1}, {"type": "32nd", "dots": 0}),
    (
        {"type": "half", "dots": 1},
        {"type": "16th", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    ({"type": "half", "dots": 1}, {"type": "16th", "dots": 0}),
    (
        {"type": "half", "dots": 1},
        {"type": "eighth", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "half", "dots": 1},
        {"type": "16th", "dots": 0},
        {"type": "16th", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "half", "dots": 2},
        {"type": "32nd", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "half", "dots": 1},
        {"type": "quarter", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "half", "dots": 2},
        {"type": "eighth", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
    (
        {"type": "half", "dots": 3},
        {"type": "16th", "dots": 0, "actual_notes": 3, "normal_notes": 2},
    ),
]


UNABBREVS = [
    (re.compile(r"(crescendo|cresc\.?)"), "crescendo"),
    (re.compile(r"(smorzando|smorz\.?)"), "smorzando"),
    (re.compile(r"(decrescendo|(decresc|decr|dimin|dim)\.?)"), "diminuendo"),
    (re.compile(r"((acceler|accel|acc)\.?)"), "accelerando"),
    (re.compile(r"(ritenente|riten\.?)"), "ritenuto"),
    (re.compile(r"((ritard|rit)\.?)"), "ritardando"),
    (re.compile(r"((rallent|rall)\.?)"), "rallentando"),
    (re.compile(r"(dolciss\.?)"), "dolcissimo"),
    (re.compile(r"((sosten|sost)\.?)"), "sostenuto"),
    (re.compile(r"(delicatiss\.?)"), "delicatissimo"),
    (re.compile(r"(leggieramente|leggiermente|leggiero|legg\.?)"), "leggiero"),
    (re.compile(r"(leggierissimo|(leggieriss\.?))"), "leggierissimo"),
    (re.compile(r"(scherz\.?)"), "scherzando"),
    (re.compile(r"(tenute|ten\.?)"), "tenuto"),
    (re.compile(r"(allegretto)"), "allegro"),
    (re.compile(r"(espress\.?)"), "espressivo"),
    (re.compile(r"(ligato)"), "legato"),
    (re.compile(r"(ligatissimo)"), "legatissimo"),
    (re.compile(r"((rinforz|rinf|rfz|rf)\.?)"), "rinforzando"),
]


TWO_PI = 2 * np.pi
SAMPLE_RATE = 44100
DTYPE = float

NATURAL_INTERVAL_RATIOS = {
    0: 1,
    1: 16 / 15,  # 15/14, 11/10
    2: 8 / 7,  # 9/8, 10/9, 12/11, 13/14
    3: 6 / 5,  # 7/6,
    4: 5 / 4,
    5: 4 / 3,
    6: 7 / 5,  # 13/9,
    7: 3 / 2,
    8: 8 / 5,
    9: 5 / 3,
    10: 7 / 4,  # 13/7
    11: 15 / 8,
    12: 2,
}

# symmetric five limit temperament with supertonic = 10:9
FIVE_LIMIT_INTERVAL_RATIOS = {
    0: 1,
    1: 16 / 15,
    2: 10 / 9,
    3: 6 / 5,
    4: 5 / 4,
    5: 4 / 3,
    6: 7 / 5,
    7: 3 / 2,
    8: 8 / 5,
    9: 5 / 3,
    10: 9 / 5,
    11: 15 / 8,
    12: 2,
}


EPSILON = 0.0001

KEYS = [
    ("C", "major", 0),
    ("Db", "major", -5),
    ("D", "major", 2),
    ("Eb", "major", -3),
    ("E", "major", 4),
    ("F", "major", -1),
    ("F#", "major", 6),
    ("G", "major", 1),
    ("Ab", "major", -4),
    ("A", "major", 3),
    ("Bb", "major", -2),
    ("B", "major", 5),
    ("C", "minor", -3),
    ("C#", "minor", 4),
    ("D", "minor", -1),
    ("D#", "minor", 6),
    ("E", "minor", 1),
    ("F", "minor", -4),
    ("F#", "minor", 3),
    ("G", "minor", -2),
    ("G#", "minor", 5),
    ("A", "minor", 0),
    ("Bb", "minor", -5),
    ("B", "minor", 2),
]

VALID_KEY_PROFILES = [
    "krumhansl_kessler",
    "kk",
    "temperley",
    "tp",
    "kostka_payne",
    "kp",
]


# Krumhansl--Kessler Key Profiles

# From Krumhansl's "Cognitive Foundations of Musical Pitch" pp.30
key_prof_maj_kk = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)

key_prof_min_kk = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Temperley Key Profiles

# CBMS (from "Music and Probability" Table 6.1, pp. 86)
key_prof_maj_cbms = np.array(
    [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
)

key_prof_min_cbms = np.array(
    [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
)

# Kostka-Payne (from "Music and Probability" Table 6.1, pp. 86)
key_prof_maj_kp = np.array(
    [0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400]
)

key_prof_min_kp = np.array(
    [0.712, 0.048, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330]
)


# Scaling factors
MAX = 9999999999999
MIN_INTERVAL = 0.01
MAX_INTERVAL = 2  # in seconds
CLUSTER_WIDTH = 1 / 12  # in seconds
N_CLUSTERS = 100
INIT_DURATION = 10  # in seconds
TIMEOUT = 10  # in seconds
TOLERANCE_POST = 0.4  # propotion of beat_interval
TOLERANCE_PRE = 0.2  # proportion of beat_interval
TOLERANCE_INNER = 1 / 12
CORRECTION_FACTOR = 1 / 4  # higher => more correction (speed changes)
MAX_AGENTS = 100  # delete low-scoring agents when there are more than MAX_AGENTS
CHORD_SPREAD_TIME = 1 / 12  # for onset aggregation


Voc_majmin = ["Cad64", "V", "viio", "V7", "N", "It", "Fr7", "Ger7", "v"]

Voc_maj_only = [
    "I",
    "ii",
    "iii",
    "IV",
    "vi",
    "I7",
    "ii7",
    "iii7",
    "IV7",
    "vi7",
    "viio7",
    "V+",
]

Voc_min_only = [
    "i",
    "iio",
    "III+",
    "iv",
    "VI",
    "i7",
    "iio7",
    "III+7",
    "iv7",
    "VI7",
    "viio7",
]

Voc_maj = Voc_majmin + Voc_maj_only
Voc_min = Voc_majmin + Voc_min_only

ACCEPTED_ROMANS = list(set(Voc_maj + Voc_min))

Voc_T_degree = [
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
]


BASE_PC = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

ALT_TO_INT = {
    "--": -2,
    "-": -1,
    "b": -1,
    "bb": -2,
    "": 0,
    "#": 1,
    "##": 2,
}

INT_TO_ALT = {
    -2: "--",
    -1: "-",
    0: "",
    1: "#",
    2: "##",
}


LOCAL_KEY_TRASPOSITIONS_DCML = {
    "minor": {
        "i": (1, "P"),
        "ii": (2, "M"),
        "iii": (3, "m"),
        "iv": (4, "P"),
        "v": (5, "P"),
        "vi": (6, "m"),
        "vii": (7, "m"),
    },
    "major": {
        "i": (1, "P"),
        "ii": (2, "M"),
        "iii": (3, "M"),
        "iv": (4, "P"),
        "v": (5, "P"),
        "vi": (6, "M"),
        "vii": (7, "M"),
    },
}

# ["G", "F", "C", "percussion", "TAB", "jianpu", "none"]
CLEF_TO_INT = {
    "G": 0,
    "F": 1,
    "C": 2,
    "percussion": 3,
    "TAB": 4,
    "jianpu": 5,
    "none": 6,
}
INT_TO_CLEF = {v: k for k, v in CLEF_TO_INT.items()}
