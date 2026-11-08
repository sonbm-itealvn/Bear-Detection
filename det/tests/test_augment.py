import numpy as np
from PIL import Image
from det.augment import hflip, small_rotate, scale_shift

def _img(): return Image.new("RGB", (640, 360), (0, 0, 0))

def test_hflip_boxes():
    im, b, ig = hflip(_img(), np.array([[10., 20., 110., 220.]]), np.zeros((0, 4)))
    np.testing.assert_allclose(b, [[530., 20., 630., 220.]])

def test_rotation_keeps_box_inside_and_close():
    rng = np.random.RandomState(0)
    b0 = np.array([[200., 100., 400., 300.]])
    im, b, ig = small_rotate(_img(), b0, np.zeros((0, 4)), rng, max_deg=7)
    assert (b[:, 0] >= 0).all() and (b[:, 2] <= 640).all() and (b[:, 3] <= 360).all()
    assert np.abs(b - b0).max() < 40

def test_scale_shift_moves_pixels_with_boxes():
    img = Image.new("RGB", (640, 360), (0, 0, 0))
    img.paste((255, 255, 255), (300, 150, 340, 190))          # white square = the "bear"
    for seed in range(5):
        im, b, _ = scale_shift(img, np.array([[300., 150., 340., 190.]]), np.zeros((0, 4)), np.random.RandomState(seed))
        if len(b) == 0:
            continue
        a = np.asarray(im)[..., 0] > 128
        ys, xs = np.nonzero(a)
        np.testing.assert_allclose([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], b[0], atol=2.5)


def _square_img(box):
    img = Image.new("RGB", (640, 360), (0, 0, 0))
    img.paste((255, 255, 255), tuple(int(v) for v in box))
    return img

def _mass_box(im):
    a = np.asarray(im)[..., 0] > 128
    ys, xs = np.nonzero(a)
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]

def test_trainer_classes_load_without_side_effects(monkeypatch):
    import os
    from det.augment import trainer_classes
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    k = trainer_classes()
    assert {"ScreenCaptureAugmentation", "NightVisionSimulation"} <= set(k)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"

def test_perspective_moves_boxes_with_pixels():
    from det.augment import perspective
    b0 = np.array([[100., 100., 200., 200.]])
    for seed in range(6):
        im, b, _ = perspective(_square_img(b0[0]), b0, np.zeros((0, 4)), np.random.RandomState(seed))
        np.testing.assert_allclose(_mass_box(im), b[0], atol=2.5)

def test_distance_shrinks_box_with_pixels():
    from det.augment import distance
    b0 = np.array([[200., 100., 400., 300.]])
    for seed in range(6):
        im, b, _ = distance(_square_img(b0[0]), b0, np.zeros((0, 4)), np.random.RandomState(seed), bg="black")
        assert (b[0, 2] - b[0, 0]) < 200 * 0.56
        np.testing.assert_allclose(_mass_box(im), b[0], atol=2.5)

def test_mosaic_places_four_items():
    from det.augment import mosaic2x2
    items = [(_square_img([300, 150, 340, 190]), np.array([[300., 150., 340., 190.]]), np.zeros((0, 4)))] * 4
    im, b, ig = mosaic2x2(items, np.random.RandomState(0))
    assert im.size == (640, 360) and len(b) == 4
    a = np.asarray(im)[..., 0] > 128
    for x1, y1, x2, y2 in b:
        assert a[int((y1 + y2) / 2), int((x1 + x2) / 2)]
