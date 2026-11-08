import numpy as np
from det.labels.homography import project_boxes, quad_is_convex

def test_identity_and_translation():
    b = np.array([[10, 20, 50, 60]], float)
    np.testing.assert_allclose(project_boxes(np.eye(3), b, 100, 100), b)
    H = np.array([[1, 0, 5], [0, 1, -3], [0, 0, 1]], float)
    np.testing.assert_allclose(project_boxes(H, b, 100, 100), [[15, 17, 55, 57]])

def test_clip():
    H = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], float)
    out = project_boxes(H, np.array([[10, 10, 80, 80]], float), 100, 100)
    assert out[0, 2] == 100 and out[0, 3] == 100

def test_convexity():
    assert quad_is_convex(np.eye(3), [0, 0, 10, 10])
    H = np.array([[1, 0, 0], [0, 1, 0], [0.2, 0, 1]], float)       # strong perspective folds the quad
    assert not quad_is_convex(H, [-20, 0, 10, 10])


def test_estimate_h_recovers_known_warp():
    import cv2
    from PIL import Image
    from det.labels.homography import estimate_h
    rng = np.random.RandomState(0)
    ref = cv2.GaussianBlur((rng.rand(540, 960, 3) * 255).astype("uint8"), (5, 5), 0)
    for _ in range(60):                                      # texture with corners for ORB
        x, y = rng.randint(0, 900), rng.randint(0, 500)
        cv2.rectangle(ref, (x, y), (x + rng.randint(10, 60), y + rng.randint(10, 60)),
                      tuple(int(c) for c in rng.randint(0, 255, 3)), -1)
    Ht = np.array([[0.9, 0.05, 60], [-0.03, 0.92, 30], [1e-5, 2e-5, 1]], float)   # ref -> frame
    frame = cv2.warpPerspective(ref, Ht, (1280, 720))
    H, inl = estimate_h(Image.fromarray(ref[:, :, ::-1]), Image.fromarray(frame[:, :, ::-1]))
    assert H is not None and inl >= 50
    b = np.array([[200., 100., 500., 400.]])
    np.testing.assert_allclose(project_boxes(H, b, 1280, 720), project_boxes(Ht, b, 1280, 720), atol=6)


def test_parse_ref_stem():
    from det.labels.homography import parse_ref_stem
    cases = {
        "cap1009_bear__bear_hard__train_00004__MFDC0234_0400.jpg": "train_00004",
        "acc1002_bear__6bgvjPZvcKc_0004__MFDC0210_0143.jpg": "6bgvjPZvcKc_0004",
        "acc1002_bear___qlckdZwy2M_0004__MFDC0084_0342.jpg": "_qlckdZwy2M_0004",
        "acc1109_images__bear__iStock_1006462858__MFDC0479_0025.jpg": "iStock_1006462858",
        "acc1809_video__bear__aAfXsxLSblM_0040__MFDC0075_0356.jpg": "aAfXsxLSblM_0040",
        "brown_bear_brown_bear_1_a.jpg": None,
    }
    for n, s in cases.items():
        assert parse_ref_stem(n) == s, (n, parse_ref_stem(n))


def test_project_row_gates_on_inliers_and_convexity():
    from det.labels.homography import project_row
    ref = {"boxes": [{"xyxy": [10, 10, 50, 50], "tier": "A"}], "ignore": [[60, 60, 70, 70]]}
    H = np.array([[2, 0, 5], [0, 2, 5], [0, 0, 1]], float)
    r = project_row(ref, H, 120, 200, 200)
    assert r["status"] == "auto" and r["boxes"][0]["xyxy"] == [25.0, 25.0, 105.0, 105.0]
    assert r["boxes"][0]["src"] == "homog" and r["ignore"] == [[125.0, 125.0, 145.0, 145.0]]
    assert project_row(ref, H, 30, 200, 200)["status"] == "masked"              # too few inliers
    assert project_row(ref, None, 0, 200, 200)["status"] == "masked"
