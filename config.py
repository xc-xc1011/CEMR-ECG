"""Global configuration for ECG heartbeat classification experiments."""

import os


# ========== Path configuration ==========
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "mitdb")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def ensure_dirs():
    """Create standard data/result directories for scripts that write outputs."""
    for d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)


# ========== Dataset configuration ==========
# AAMI EC57 inter-patient protocol, excluding paced records 102/104/107/217.
DS1_RECORDS = [
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
    201, 203, 205, 207, 208, 209, 215, 220, 223, 230,
]
DS2_RECORDS = [
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
    213, 214, 219, 221, 222, 228, 231, 232, 233, 234,
]

FS = 360
PRE_R = 90
POST_R = 144
WIN_LEN = PRE_R + POST_R  # 234

AAMI_MAP = {
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,
    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,
    "V": 2,
    "E": 2,
    "F": 3,
    "/": 4,
    "f": 4,
    "Q": 4,
}
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
N_CLASSES = 5


# ========== Signal preprocessing ==========
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0
BANDPASS_ORDER = 4


# ========== Experiment configuration ==========
RANDOM_SEED = 42
SEEDS = [303, 1303, 2303, 3303, 4303]
