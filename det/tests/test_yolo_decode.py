import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import numpy as np
import torch

def _model():
    from det.probe_yolo import build, RawHead
    m = build("yolov8n")
    return m, RawHead(m).eval()

def test_decode_raw_matches_ultralytics():
    from det.yolo.decode import decode_raw
    m, raw = _model()
    x = torch.randn(1, 3, 384, 640)
    with torch.no_grad():
        outs = [o.numpy()[0] for o in raw(x)]
        y = m(x)[0][0].numpy()                      # (4+nc, N): cx, cy, w, h (input px), sigmoid score
    boxes, scores = decode_raw(outs)
    ref = np.stack([y[0] - y[2] / 2, y[1] - y[3] / 2, y[0] + y[2] / 2, y[1] + y[3] / 2], 1)
    assert boxes.shape == ref.shape
    assert np.abs(boxes - ref).max() < 1e-2
    assert np.abs(scores - y[4]).max() < 1e-5

def test_letterbox_384():
    from det.yolo.geometry import letterbox_params
    assert letterbox_params(1920, 1080) == (640, 360, 0, 12)
    assert letterbox_params(640, 384) == (640, 384, 0, 0)

def test_decode_maps_back_and_nms():
    from det.yolo.decode import decode
    # two anchors at P3 cell (10,20) and (10,21) predicting the same box, one far away
    outs = []
    for (h, w) in ((48, 80), (24, 40), (12, 20)):
        box = np.zeros((64, h, w), np.float32); cls = np.full((1, h, w), -10.0, np.float32)
        outs += [box, cls]
    def put(level, i, j, z, d):                     # d = distance in stride units per side, as a one-hot DFL bin
        box, cls = outs[2 * level], outs[2 * level + 1]
        cls[0, i, j] = z
        for s in range(4):
            box[s * 16:(s + 1) * 16, i, j] = -20.0; box[s * 16 + d, i, j] = 20.0
    put(0, 10, 20, 3.0, 4); put(0, 10, 21, 2.0, 4); put(2, 5, 15, 3.0, 2)
    b = decode(outs, 1920, 1080, thr=0.3)
    assert len(b) == 2
    # P3 cell (10,20): centre (20.5*8, 10.5*8) = (164, 84) in net px, +-32 px; net->orig: (x-0)/(1/3), (y-12)/(1/3)
    np.testing.assert_allclose(b[0, :4], [(164 - 32) * 3, (84 - 32 - 12) * 3, (164 + 32) * 3, (84 + 32 - 12) * 3], atol=0.5)


def test_fallback_skips_degenerate_boxes_in_bar():
    from det.yolo.decode import decode
    outs = []
    for (h, w) in ((48, 80), (24, 40), (12, 20)):
        outs += [np.full((64, h, w), 0.0, np.float32), np.full((1, h, w), -9.0, np.float32)]
    def put(i, j, z, d):
        box, cls = outs[0], outs[1]
        cls[0, i, j] = z
        for s in range(4):
            box[s * 16:(s + 1) * 16, i, j] = -20.0; box[s * 16 + d, i, j] = 20.0
    put(20, 2, -1.0, 0)          # strongest, but a 0-size box inside the side bar of a portrait
    put(20, 40, -2.0, 3)         # weaker, valid box in the content area
    b = decode(outs, 500, 800, thr=0.3, fallback=True)
    assert len(b) == 1 and b[0, 2] - b[0, 0] >= 1
