"""Box-aware augmentations on the LETTERBOXED 640x360 image. Every op takes and returns
(PIL image, boxes [N,4], ignore [M,4]) in letterbox pixels; boxes that leave the frame
are clipped and dropped below 2 px."""
import numpy as np
from PIL import Image

W, H = 640, 360


def _clip(b):
    b = np.asarray(b, float).reshape(-1, 4).copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
    return b[(b[:, 2] - b[:, 0] >= 2) & (b[:, 3] - b[:, 1] >= 2)]


def _flip_boxes(b):
    b = np.asarray(b, float).reshape(-1, 4)
    return np.stack([W - b[:, 2], b[:, 1], W - b[:, 0], b[:, 3]], 1) if len(b) else b


def hflip(img, boxes, ignore):
    return img.transpose(Image.FLIP_LEFT_RIGHT), _flip_boxes(boxes), _flip_boxes(ignore)


def _affine_boxes(M, b):
    b = np.asarray(b, float).reshape(-1, 4)
    if not len(b):
        return b
    pts = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], 1)      # N,4,2
    q = pts @ M[:, :2].T + M[:, 2]
    return _clip(np.concatenate([q.min(1), q.max(1)], 1))


def _warp(img, M):
    inv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
    return img.transform((W, H), Image.AFFINE, inv.flatten().tolist(), Image.BILINEAR, fillcolor=(0, 0, 0))


def apply_affine(img, boxes, ignore, M):
    return _warp(img, M), _affine_boxes(M, boxes), _affine_boxes(M, ignore)


def scale_shift(img, boxes, ignore, rng, smin=0.6, smax=1.4):
    s = rng.uniform(smin, smax)
    if s > 1:
        tx, ty = rng.uniform((1 - s) * W, 0), rng.uniform((1 - s) * H, 0)
    else:
        tx, ty = rng.uniform(0, (1 - s) * W), rng.uniform(0, (1 - s) * H)
    return apply_affine(img, boxes, ignore, np.array([[s, 0, tx], [0, s, ty]]))


def small_rotate(img, boxes, ignore, rng, max_deg=7):
    a = np.deg2rad(rng.uniform(-max_deg, max_deg))
    c, s = np.cos(a), np.sin(a)
    cx, cy = W / 2, H / 2
    M = np.array([[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy]])
    return apply_affine(img, boxes, ignore, M)


# ---------------------------------------------------------------- trainer augmentations
TRAINER = "/data/nas07_new/PersonalData/phuocthien/train_mobilenetv2_relu_novaic_23_4.py"
_CLASSES = None


class _NoPerspective:
    """`random` stand-in for ScreenCaptureAugmentation: identical except that the
    'perspective' effect is never drawn (it is applied box-aware by perspective())."""
    def __init__(self):
        import random as _r
        self._r = _r

    def __getattr__(self, k):
        return getattr(self._r, k)

    def sample(self, pop, k):
        got = [e for e in self._r.sample(pop, k) if e != "perspective"]
        return got or ["moire"]


def trainer_classes():
    """The trainer's photometric augmentation classes, loaded by AST without executing the
    trainer module (its top level sets CUDA_VISIBLE_DEVICES and AWS credentials)."""
    global _CLASSES
    if _CLASSES is None:
        import ast, random
        import cv2
        from PIL import ImageEnhance, ImageFilter
        tree = ast.parse(open(TRAINER).read())
        want = {"ScreenCaptureAugmentation", "NightVisionSimulation"}
        _CLASSES = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in want:
                ns = {"np": np, "cv2": cv2, "Image": Image, "ImageFilter": ImageFilter,
                      "ImageEnhance": ImageEnhance,
                      "random": _NoPerspective() if node.name == "ScreenCaptureAugmentation" else random}
                exec(compile(ast.Module(body=[node], type_ignores=[]), TRAINER, "exec"), ns)
                _CLASSES[node.name] = ns[node.name]
    return _CLASSES


def perspective(img, boxes, ignore, rng):
    """The trainer's screen 'perspective' effect (one edge pinched 5-25 px), box-aware."""
    import cv2
    from det.labels.homography import project_boxes
    pts1 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
    pts2 = pts1.copy()
    shift = rng.randint(5, 26)
    side = rng.choice(["left", "right", "top", "bottom"])
    if side == "left":
        pts2[0][1] += shift; pts2[2][1] -= shift
    elif side == "right":
        pts2[1][1] += shift; pts2[3][1] -= shift
    elif side == "top":
        pts2[0][0] += shift; pts2[1][0] -= shift
    else:
        pts2[2][0] += shift; pts2[3][0] -= shift
    M = cv2.getPerspectiveTransform(pts1, pts2)
    out = cv2.warpPerspective(np.asarray(img), M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    pb = lambda b: _clip(project_boxes(M, b, W, H)) if len(np.asarray(b).reshape(-1, 4)) else np.zeros((0, 4))
    return Image.fromarray(out), pb(boxes), pb(ignore)


def screen_capture(img, boxes, ignore, rng, p_perspective=0.4):
    """Trainer ScreenCaptureAugmentation (photometric effects) + box-aware perspective."""
    img = trainer_classes()["ScreenCaptureAugmentation"](p=1.0)(img)
    if rng.rand() < p_perspective:
        img, boxes, ignore = perspective(img, boxes, ignore, rng)
    return img, boxes, ignore


def night_vision(img, boxes, ignore, rng=None):
    """Trainer NightVisionSimulation; purely photometric, boxes unchanged."""
    return trainer_classes()["NightVisionSimulation"](p=1.0)(img), boxes, ignore


def distance(img, boxes, ignore, rng, smin=0.2, smax=0.55, bg=None):
    """Box-aware DistanceSimulation: shrink the frame and paste it on a bear-free
    background (blurred corner crop, noise or black), as the trainer does."""
    from PIL import ImageFilter
    s = rng.uniform(smin, smax)
    nw, nh = int(W * s), int(H * s)
    small = img.resize((nw, nh), Image.BILINEAR)
    bg = bg or rng.choice(["corner_soft", "corner_heavy", "noise", "black"], p=[0.5, 0.3, 0.15, 0.05])
    if bg == "black":
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
    elif bg == "noise":
        canvas = Image.fromarray(rng.randint(0, 50, (H, W, 3)).astype(np.uint8))
    else:
        cw, ch = W // 4, H // 4
        x0, y0 = [(0, 0), (W - cw, 0), (0, H - ch), (W - cw, H - ch)][rng.randint(4)]
        canvas = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((W, H), Image.BILINEAR)
        canvas = canvas.filter(ImageFilter.GaussianBlur(15 if bg == "corner_soft" else 30))
        if bg == "corner_heavy":
            canvas = Image.fromarray((np.asarray(canvas, np.float32) * rng.uniform(0.5, 0.85)).astype(np.uint8))
    px, py = rng.randint(0, W - nw + 1), rng.randint(0, H - nh + 1)
    canvas.paste(small, (px, py))
    M = np.array([[nw / W, 0, px], [0, nh / H, py]])
    return canvas, _affine_boxes(M, boxes), _affine_boxes(M, ignore)


def mosaic2x2(items, rng):
    """Four (img, boxes, ignore) items, each shrunk to a 320x180 quadrant."""
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    bs, igs = [], []
    order = rng.permutation(4)
    for q, k in enumerate(order):
        img, b, ig = items[k]
        ox, oy = (q % 2) * W // 2, (q // 2) * H // 2
        canvas.paste(img.resize((W // 2, H // 2), Image.BILINEAR), (ox, oy))
        M = np.array([[0.5, 0, ox], [0, 0.5, oy]])
        bs.append(_affine_boxes(M, b)); igs.append(_affine_boxes(M, ig))
    return canvas, np.concatenate(bs).reshape(-1, 4), np.concatenate(igs).reshape(-1, 4)
