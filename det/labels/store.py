"""Split-independent box-label store: one JSON line per image, keyed by sha1 of the file
bytes (dhash fallback). Survives resplits/quarantines; a changed class folder marks stale."""
import hashlib, json, os, time
from pathlib import Path
import numpy as np
from PIL import Image


def load_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path):
    g = np.asarray(load_rgb(path).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return "%016x" % int("".join("1" if b else "0" for b in bits), 2)


class LabelStore:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = {}
        self.stale = []
        if self.path.exists():
            for line in open(self.path):
                if line.strip():
                    r = json.loads(line)
                    self.rows[r["sha1"]] = r
        self._by_dhash = {r["dhash"]: r for r in self.rows.values() if r.get("dhash")}

    def put(self, row):
        row.setdefault("ts", time.time())
        self.rows[row["sha1"]] = row
        if row.get("dhash"):
            self._by_dhash[row["dhash"]] = row

    def get(self, sha1):
        return self.rows.get(sha1)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for k in sorted(self.rows):
                f.write(json.dumps(self.rows[k]) + "\n")
        os.replace(tmp, self.path)

    def join(self, paths):
        self.stale = []
        out = {}
        for p in map(Path, paths):
            r = self.rows.get(sha1_file(p))
            if r is None:
                r = self._by_dhash.get(dhash(p))
            if r is not None and r.get("cls") != p.parent.name:
                self.stale.append(str(p))
                r = None
            out[str(p)] = r
        return out
