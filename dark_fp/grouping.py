import numpy as np

_NEG_KEYS = ('dark_dog', 'dark_fox', 'domestic_dog', 'red_fox', '_fox', 'wild_boar',
             'coyote', 'bobcat', 'pallas', 'macaque', 'mfdc', 'pasteaug_neg')
_POS_KEYS = ('night_vision', 'night', 'low_light', 'nightbear', '_ir_', 'infrared',
             'pasteaug_pos')

def classify_group(path):
    p = path.lower()
    if '/non_bear/' in p and any(k in p for k in _NEG_KEYS):
        return 'dark_quad_negative'
    if '/bear/' in p and any(k in p for k in _POS_KEYS):
        return 'night_bear_positive'
    return 'other'

def apply_group_weights(paths, base_weights, dark_quad_boost=3.0, night_bear_boost=2.0):
    w = np.asarray(base_weights, dtype=np.float64).copy()
    for i, p in enumerate(paths):
        g = classify_group(str(p))
        if g == 'dark_quad_negative':
            w[i] *= dark_quad_boost
        elif g == 'night_bear_positive':
            w[i] *= night_bear_boost
    return w
