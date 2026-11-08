from det.labels.domains import domain_of

def test_domains():
    cases = {
        "bear_pasteaug_pos_00012.jpg": "paste",
        "acc1109_bear__x__frame_0001.jpg": "screen",
        "cap1009_bear__bear_ir__train_01824__MFDC0001_0036.jpg": "screen",
        "bear_full_capture_seg0227_sub00.jpg": "screen",
        "bear_MFDC0086_48.jpg": "screen",
        "distance_00123.jpg": "small",
        "polar_bear_polar_bear_27935205_477e6297.jpg": "rare",
        "giant_panda_giant_panda_198887157_ca615ff2.jpg": "rare",
        "kermode_bear_kermode_bear_1_a.jpg": "rare",
        "brown_bear_brown_bear_25406887_b6b88f28.jpg": "inat",
        "asian_black_bear_asian_black_bear_182167218_8a82d3ff.jpg": "inat",
        "train_bear1_04513.jpg": "ir",
        "bear_frames_night_vision_v2_gpu5_00701_conf0.82.jpg": "ir",
        "low_light_gpu4_00012.jpg": "ir",
        "BERA_IR_0001.jpg": "ir",
        "bear_frames_in_urban_gpu4_00244_conf0.70.jpg": "youtube",
        "motion_blur_gpu5_00012.jpg": "youtube",
        "Video_12.jpg": "youtube",
        "something_else.jpg": "other",
    }
    for f, d in cases.items():
        assert domain_of(f) == d, (f, domain_of(f), d)
