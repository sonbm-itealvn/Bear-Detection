import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import numpy as np
from PIL import Image

def test_output_keys():
    from det.yolo.chip import output_keys
    assert output_keys()["box_p3"] == "Conv_box_p3_Y" and len(output_keys()) == 6

def test_stage_384_letterbox(tmp_path):
    from det.yolo.chip import letterbox_384_jpeg
    src = tmp_path / "a.jpg"; Image.new("RGB", (1920, 1080), (255, 255, 255)).save(src)
    dst = tmp_path / "b.jpg"; letterbox_384_jpeg(src, dst)
    a = np.asarray(Image.open(dst))
    assert a.shape == (384, 640, 3) and a[:10].max() < 20 and a[200, 320].min() > 235
