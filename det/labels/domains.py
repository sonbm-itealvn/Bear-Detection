"""Source domain of a dataset image from its file name (rules from the 15/10 survey of
train/bear). Order matters: first match wins."""
import re

RARE = ("polar_bear_", "giant_panda_", "spectacled_bear_", "sun_bear_", "sloth_bear_", "kermode_bear_")
RULES = [
    ("paste", re.compile(r"^bear_pasteaug_")),
    ("screen", re.compile(r"^(acc\d{4}_|cap1009_|bear_full_capture_seg|bear_MFDC|bear_capture_)")),
    ("small", re.compile(r"^distance_")),
    ("rare", re.compile(r"^(" + "|".join(RARE) + ")")),
    ("inat", re.compile(r"^[a-z_]+_bear_[a-z_]+_bear_\d")),
    ("ir", re.compile(r"(^(train|val|test)_bear\d*_|night_vision|^low_light_|^IR_gpu|^BERA_IR|^Nightbear|trail_cam)")),
    ("youtube", re.compile(r"(_gpu\d|^Video|^Bear_JapNews)")),
]
DOMAINS_EVAL = ["inat", "rare", "ir", "youtube", "small", "screen"]


def domain_of(filename):
    name = filename.rsplit("/", 1)[-1]
    for d, rx in RULES:
        if rx.search(name):
            return d
    return "other"
