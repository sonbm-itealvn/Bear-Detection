"""Detection dataset over bear_dataset_v_final_final/<split>/{bear,non_bear}, joined to
the label store by sha1 (cached by path+size+mtime so later runs do not rehash 200k files).
Images are letterboxed exactly like the classifier's ResizeWithPadding and normalised
like NoviacTransform: BGR, (x - [104,116,124]) / 255."""
import json, os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from det import augment as A
from det.geometry import IN_W, IN_H, letterbox_params, orig_to_lb
from det.labels.store import LabelStore, load_rgb, sha1_file
from det.targets import encode

MEAN_BGR = np.array([104.0, 116.0, 124.0], np.float32)


def letterbox(img):
    nw, nh, px, py = letterbox_params(*img.size)
    canvas = Image.new("RGB", (IN_W, IN_H), (0, 0, 0))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    return canvas


def to_tensor(img):
    a = np.asarray(img, np.float32)[:, :, ::-1]                 # RGB -> BGR
    return torch.from_numpy(((a - MEAN_BGR) / 255.0).transpose(2, 0, 1).copy())


def _sha1_cached(paths, cache_path):
    cache = {}
    if cache_path and Path(cache_path).exists():
        cache = json.load(open(cache_path))
    out, dirty = {}, False
    for p in paths:
        st = os.stat(p)
        key = str(p)
        c = cache.get(key)
        if c and c[0] == st.st_size and c[1] == int(st.st_mtime):
            out[key] = c[2]
        else:
            out[key] = sha1_file(p)
            cache[key] = [st.st_size, int(st.st_mtime), out[key]]
            dirty = True
    if cache_path and dirty:
        tmp = Path(f"{cache_path}.{os.getpid()}.tmp")       # per process: trainers run concurrently
        json.dump(cache, open(tmp, "w"))
        os.replace(tmp, cache_path)
    return out


class DetDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, stride, train, store_path, cache_path=None,
                 exclude_sha1=(), only_sha1=None, limit=0, epochs=24, seed=0):
        self.stride, self.train, self.epochs = stride, train, epochs
        self.epoch = 0
        store = LabelStore(store_path)
        paths = []
        for cls in ("bear", "non_bear"):
            d = Path(root) / split / cls
            paths += [d / f for f in sorted(os.listdir(d))]
        sha = _sha1_cached(paths, cache_path)
        exclude_sha1 = set(exclude_sha1)
        self.meta, self.n_stale, self.n_unlabelled = [], 0, 0
        for p in paths:
            s, cls = sha[str(p)], p.parent.name
            if s in exclude_sha1 or (only_sha1 is not None and s not in only_sha1):
                continue
            r = store.get(s)
            if r is not None and r.get("cls") != cls:
                self.n_stale += 1
                continue
            if r is not None and r.get("status") == "excluded":
                continue
            if r is None:
                self.n_unlabelled += 1
                if cls == "bear":                      # unlabelled bear: masked, never background
                    r = {"boxes": [], "ignore": [], "status": "masked", "domain": "other"}
                else:
                    r = {"boxes": [], "ignore": [], "status": "auto", "domain": "other"}
            self.meta.append({"path": str(p), "cls": cls, "sha1": s, "domain": r.get("domain", "other"),
                              "boxes": [b["xyxy"] for b in r["boxes"]], "ignore": r["ignore"],
                              "masked": r["status"] == "masked" and cls == "bear"})
        if limit:
            idx = np.random.RandomState(seed).permutation(len(self.meta))[:limit]
            self.meta = [self.meta[i] for i in sorted(idx)]

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.meta)

    def _load(self, i):
        m = self.meta[i]
        img = load_rgb(m["path"])
        w, h = img.size
        b = orig_to_lb(np.asarray(m["boxes"], float).reshape(-1, 4), w, h)
        ig = orig_to_lb(np.asarray(m["ignore"], float).reshape(-1, 4), w, h)
        return letterbox(img), b, ig, (w, h)

    def _rng(self, i):
        """Fresh stream per call: DataLoader workers are forked copies, so a shared
        RandomState would give every worker the same augmentation sequence."""
        self._calls = getattr(self, "_calls", 0) + 1
        return np.random.RandomState((torch.initial_seed() + 7919 * i + 104729 * self._calls
                                      + 15485863 * self.epoch) % 2 ** 32)

    def _tile(self, j, img=None, b=None, ig=None):
        """One mosaic tile; a masked bear tile becomes a whole-tile ignore region."""
        if img is None:
            img, b, ig, _ = self._load(j)
        if self.meta[j]["masked"]:
            ig = np.concatenate([np.asarray(ig, float).reshape(-1, 4), [[0.0, 0.0, A.W, A.H]]])
        return img, b, ig

    def _mosaic(self, i, others, rng, img=None, b=None, ig=None):
        tiles = [self._tile(i, img, b, ig) if img is not None else self._tile(i)] + [self._tile(int(j)) for j in others]
        img, b, ig = A.mosaic2x2(tiles, rng)
        return img, b, ig, False          # masking now lives in the tiles' ignore regions

    def _augment(self, i, img, b, ig):
        rng, m = self._rng(i), self.meta[i]
        bear = m["cls"] == "bear"
        masked = m["masked"]
        if rng.rand() < 0.25 and self.epoch < 0.8 * self.epochs:
            img, b, ig, masked = self._mosaic(i, rng.randint(0, len(self), 3), rng, img, b, ig)
        if rng.rand() < 0.5:
            img, b, ig = A.hflip(img, b, ig)
        if bear and rng.rand() < 0.15:
            img, b, ig = A.distance(img, b, ig, rng)
        elif rng.rand() < 0.5:
            img, b, ig = A.scale_shift(img, b, ig, rng)
        if rng.rand() < 0.3:
            img, b, ig = A.small_rotate(img, b, ig, rng)
        if rng.rand() < 0.3:                           # symmetric on both classes
            img, b, ig = A.screen_capture(img, b, ig, rng)
        if bear and rng.rand() < 0.15:
            img, b, ig = A.night_vision(img, b, ig)
        return img, b, ig, masked

    def __getitem__(self, i):
        m = self.meta[i]
        img, b, ig, (w, h) = self._load(i)
        masked = m["masked"]
        if self.train:
            img, b, ig, masked = self._augment(i, img, b, ig)
        t = encode(b, ig, self.stride, masked=masked)
        t = {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in t.items()}
        return to_tensor(img), t, {"path": m["path"], "w": w, "h": h, "boxes_lb": b,
                                   "masked": m["masked"], "domain": m["domain"]}


def collate(batch):
    x = torch.stack([b[0] for b in batch])
    keys = [k for k in batch[0][1] if k != "n_pos"]
    t = {k: torch.stack([b[1][k] for b in batch]) for k in keys}
    return x, t, [b[2] for b in batch]


class DetSampler(torch.utils.data.Sampler):
    """Batches of 2/3 bear and 1/3 non_bear; half of the non_bear drawn from `hard`."""
    def __init__(self, ds, batch=32, n_batches=1000, hard=(), bear_frac=2 / 3, hard_frac=0.5, seed=0):
        self.batch, self.n_batches, self.seed = batch, n_batches, seed
        self.bear = [i for i, m in enumerate(ds.meta) if m["cls"] == "bear"]
        nb = [i for i, m in enumerate(ds.meta) if m["cls"] == "non_bear"]
        hard = set(hard)
        self.hard = [i for i in nb if ds.meta[i]["path"] in hard] or nb
        self.easy = [i for i in nb if ds.meta[i]["path"] not in hard] or nb
        self.n_bear = int(round(batch * bear_frac))
        self.n_hard = int(round((batch - self.n_bear) * hard_frac))
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.batch * self.n_batches

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        for _ in range(self.n_batches):
            idx = list(rng.choice(self.bear, self.n_bear)) + list(rng.choice(self.hard, self.n_hard)) + \
                list(rng.choice(self.easy, self.batch - self.n_bear - self.n_hard))
            rng.shuffle(idx)
            yield from (int(i) for i in idx)
