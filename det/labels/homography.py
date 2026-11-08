"""Camera-filmed frames (acc*, cap1009_*) show a reference image on a screen. Their boxes
are the reference image's boxes projected through the ORB+RANSAC homography that matched
the frame to its reference (parameters copied from match_acceptance_frames.py)."""
import argparse, csv, json, os, re, sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")
REF_ROOTS = [ROOT / "capture_package_1006/train", ROOT / "acceptance_merged", ROOT / "video_categorized_grouped"]

WORK = 480, 270          # match_acceptance_frames.WORK
N_FEATURES = 800         # match_acceptance_frames.N_FEATURES
RATIO = 0.75             # Lowe ratio, match_acceptance_frames.RATIO
RANSAC_PX = 5.0          # match_acceptance_frames._score
MIN_INLIERS = 50         # spec §2: below this the frame goes to review


def project_boxes(H, boxes, frame_w, frame_h):
    """Map each box's 4 corners through H, take the enclosing box, clip to the frame."""
    out = []
    for x1, y1, x2, y2 in np.asarray(boxes, float).reshape(-1, 4):
        pts = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], float).T
        q = H @ pts
        q = q[:2] / q[2:3]
        out.append([min(max(q[0].min(), 0), frame_w), min(max(q[1].min(), 0), frame_h),
                    min(max(q[0].max(), 0), frame_w), min(max(q[1].max(), 0), frame_h)])
    return np.array(out, float).reshape(-1, 4)


def quad_is_convex(H, box):
    x1, y1, x2, y2 = box
    p = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], float).T
    q = H @ p
    if (q[2] <= 0).any():                      # a corner behind the camera: folded quad
        return False
    q = (q[:2] / q[2:3]).T
    cross = [np.cross(q[(i + 1) % 4] - q[i], q[(i + 2) % 4] - q[(i + 1) % 4]) for i in range(4)]
    return all(c > 0 for c in cross) or all(c < 0 for c in cross)


def _describe(img):
    g = cv2.resize(np.asarray(img.convert("L")), WORK)
    kp, des = cv2.ORB_create(nfeatures=N_FEATURES).detectAndCompute(g, None)
    if des is None or len(kp) < 10:
        return None, None
    return np.float32([p.pt for p in kp]), des


def estimate_h(ref_img, frame_img):
    """Homography mapping ORIGINAL reference pixels to ORIGINAL frame pixels, and the
    RANSAC inlier count. (None, n) when it cannot be estimated."""
    rp, rd = _describe(ref_img)
    fp, fd = _describe(frame_img)
    if rd is None or fd is None:
        return None, 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(rd, fd, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < RATIO * b.distance]
    if len(good) < 8:
        return None, len(good)
    src = rp[[a.queryIdx for a in good]].reshape(-1, 1, 2)
    dst = fp[[a.trainIdx for a in good]].reshape(-1, 1, 2)
    Hw, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if Hw is None:
        return None, 0
    s_ref = np.diag([WORK[0] / ref_img.width, WORK[1] / ref_img.height, 1.0])
    s_frm = np.diag([WORK[0] / frame_img.width, WORK[1] / frame_img.height, 1.0])
    return np.linalg.inv(s_frm) @ Hw @ s_ref, int(mask.sum())


_STEM = re.compile(r"^(?:acc\d{4}|cap1009)_.*?bear__(.+)__MFDC\d+_\d+\.\w+$")


def parse_ref_stem(name):
    """Stem of the displayed reference image inside a camera-frame file name, e.g.
    cap1009_bear__bear_hard__train_00004__MFDC0234_0400.jpg -> train_00004."""
    m = _STEM.match(name)
    return m.group(1).split("__")[-1] if m else None


def project_row(ref_row, H, inliers, frame_w, frame_h):
    """Frame row fields from the reference row's boxes. 'auto' only when the homography
    is trusted (>= MIN_INLIERS and every box quad convex); otherwise 'masked' for review."""
    ok = H is not None and inliers >= MIN_INLIERS and all(quad_is_convex(H, b["xyxy"]) for b in ref_row["boxes"])
    if not ok:
        return {"status": "masked", "boxes": [], "ignore": [], "n_det": 0, "inliers": int(inliers)}
    boxes = []
    for b in ref_row["boxes"]:
        x = project_boxes(H, [b["xyxy"]], frame_w, frame_h)[0]
        if x[2] - x[0] >= 2 and x[3] - x[1] >= 2:
            boxes.append({**b, "xyxy": [float(v) for v in x], "src": "homog",
                          "truncated": bool(x[0] <= 2 or x[1] <= 2 or x[2] >= frame_w - 2 or x[3] >= frame_h - 2)})
    ig = [[float(v) for v in project_boxes(H, [g], frame_w, frame_h)[0]] for g in ref_row["ignore"]]
    return {"status": "auto" if boxes else "masked", "boxes": boxes, "ignore": ig,
            "n_det": len(boxes), "inliers": int(inliers)}


def _ref_index():
    idx = {}
    for root in REF_ROOTS:
        for dp, _, fs in os.walk(root):
            if root.name != "train" and Path(dp).name != "bear":
                continue                                  # only displayed BEAR references
            for f in fs:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    idx.setdefault(Path(f).stem, []).append(Path(dp) / f)
    return idx


def main(argv=None):
    """--stage list: write the reference images of every camera-filmed bear frame
    (feed it to run_detectors --files). --stage project: fuse the reference proposals and
    project them into each frame; untrusted frames are masked and listed for review."""
    from PIL import Image
    from det.labels.build_labels import build_rows
    from det.labels.store import LabelStore, dhash, load_rgb, sha1_file
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["list", "project"])
    ap.add_argument("--store", default=str(ROOT / "det_labels/labels_v1.jsonl"))
    ap.add_argument("--refs", default=str(ROOT / "det_labels/screen_refs.txt"))
    ap.add_argument("--ref-proposals", default=str(ROOT / "det_labels/proposals_screen_refs.jsonl"))
    ap.add_argument("--frames", help="restrict to these frame paths (one per line)")
    ap.add_argument("--review-list", default=str(ROOT / "det_labels/review_screen.txt"))
    a = ap.parse_args(argv)
    frames = []
    for sp in ("train", "val", "test"):
        d = ROOT / "bear_dataset_v_final_final" / sp / "bear"
        frames += [d / f for f in sorted(os.listdir(d)) if parse_ref_stem(f)]
    if a.frames:
        keep = {l.strip() for l in open(a.frames) if l.strip()}
        frames = [f for f in frames if str(f) in keep or str(f.relative_to(ROOT)) in keep]
    idx = _ref_index()
    cand = {f: idx.get(parse_ref_stem(f.name), []) for f in frames}
    if a.stage == "list":
        refs = sorted({str(p) for c in cand.values() for p in c})
        Path(a.refs).write_text("\n".join(refs) + "\n")
        print(f"{len(frames)} frames, {sum(1 for c in cand.values() if not c)} without a reference, "
              f"{len(refs)} reference images -> {a.refs}")
        return
    refrows = {}
    for line in open(a.ref_proposals):
        p = json.loads(line)
        p = {**p, "path": str(Path("/bear") / Path(p["path"]).name)}       # references are bears
        refrows[json.loads(line)["path"]] = build_rows(p)
    st = LabelStore(a.store)
    review, stats = [], {"auto": 0, "masked": 0, "no_ref": 0}
    for f, refs in cand.items():
        sha = sha1_file(f)
        old = st.get(sha)
        if old and old.get("status") in ("reviewed", "corrected", "excluded"):
            continue
        frame = load_rgb(f)
        best = (None, 0, None)
        for r in refs:
            if str(r) not in refrows:
                continue
            H, n = estimate_h(load_rgb(r), frame)
            if n > best[1]:
                best = (H, n, r)
        if best[2] is None:
            stats["no_ref"] += 1
            row = {"status": "masked", "boxes": [], "ignore": [], "n_det": 0, "inliers": 0}
        else:
            row = project_row(refrows[str(best[2])], best[0], best[1], frame.width, frame.height)
        stats[row["status"]] = stats.get(row["status"], 0) + 1
        if row["status"] == "masked":
            review.append(str(f))
        st.put({"sha1": sha, "dhash": dhash(f), "w": frame.width, "h": frame.height, "path_hint": str(f),
                "cls": "bear", "domain": "screen", "version": 1, "reviewer": None,
                "ref": str(best[2]) if best[2] else None, **row})
    st.save()
    Path(a.review_list).write_text("\n".join(review) + "\n")
    print(f"{len(frames)} frames: {stats}; {len(review)} listed in {a.review_list}")


if __name__ == "__main__":
    main()
