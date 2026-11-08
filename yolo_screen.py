"""Score images by "is there an animal in here at all", to spare the LLM judges.

Gemini through `agy` runs about two minutes per call whatever the batch size,
and the calls share one global concurrency limit -- starting a second job
halved the first one's rate rather than adding to it. At that speed the 19,401
images waiting cost well over a day. YOLOv8 does the same triage on one GPU in
minutes.

This is a TRIAGE score, not a verdict, and it is aimed at one specific failure
in each direction:

  bear species   iNaturalist casual-grade observations are full of bear SIGN --
                 scat, claw marks, stripped bark, footprints, an empty zoo
                 enclosure. Twelve rejected photos inspected by eye were all of
                 that kind. Those frames contain no animal, so YOLO sees
                 nothing, and only the no-detection images need an LLM.

  empty scenes   the harvest already screened with YOLO at 0.10 and a bear
                 close-up filling the frame still got through, because an
                 extreme close-up has no silhouette. So a detection here is
                 not the interesting signal -- absence is, and those images
                 still need the LLM. Run this only to confirm the harvest.

`--calibrate` checks the score against whatever Gemini verdicts already exist
before any threshold is used. The probe is never allowed to set its own cut.

    python yolo_screen.py --dir bear_species_1809 --gpu 2
    python yolo_screen.py --dir bear_species_1809 --calibrate
"""
import argparse
import json
from pathlib import Path

BASE = Path("/data/nas07_new/PersonalData/phuocthien")

# COCO classes that are animals. `person` (0) is deliberately NOT here: a
# keeper holding a cub and a hiker beside a scat pile look the same to a
# person-detector, so it separates nothing.
ANIMAL = set(range(14, 24))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--mode", default="bear_visible")
    a = ap.parse_args()

    root = BASE / a.dir
    out_path = BASE / f"yolo_{Path(a.dir).name}.jsonl"

    if a.calibrate:
        sc = {}
        for l in open(out_path):
            if l.strip():
                r = json.loads(l)
                sc[r["file"]] = r["animal_conf"]
        cache = BASE / f"verify_{Path(a.dir).name}_{a.mode}.jsonl"
        gem = {}
        for l in open(cache):
            if l.strip():
                r = json.loads(l)
                if r["judge"] == "gemini" and r["verdict"] in ("yes", "no"):
                    gem[r["file"]] = r["verdict"]
        both = [(f, sc[f], gem[f]) for f in gem if f in sc]
        print(f"  {len(both)} ảnh có cả điểm YOLO và phán quyết Gemini")
        if not both:
            return
        yes = sum(1 for x in both if x[2] == "yes")
        print(f"  Gemini: {yes} có gấu, {len(both)-yes} không\n")
        print(f"{'ngưỡng':>8}{'tự nhận (>=)':>14}{'trong đó Gemini nói KHÔNG':>28}"
              f"{'còn phải hỏi LLM':>19}{'trong đó thật sự CÓ':>21}")
        for t in (0.05, 0.10, 0.25, 0.40, 0.60):
            auto = [x for x in both if x[1] >= t]
            bad = [x for x in auto if x[2] == "no"]
            rest = [x for x in both if x[1] < t]
            miss = [x for x in rest if x[2] == "yes"]
            print(f"{t:>8.2f}{len(auto):>14}{len(bad):>28}{len(rest):>19}{len(miss):>21}")
        return

    import torch
    from ultralytics import YOLO
    model = YOLO(str(BASE / "yolov8m.pt"))
    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"

    files = sorted(p for p in root.rglob("*.jpg"))
    done = set()
    if out_path.exists():
        done = {json.loads(l)["file"] for l in open(out_path) if l.strip()}
    todo = [p for p in files if str(p.relative_to(root)) not in done]
    print(f"  {len(files)} ảnh, đã có {len(done)}, cần chấm {len(todo)}", flush=True)

    fh = open(out_path, "a")
    fails = 0
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        try:
            # imgsz pinned: iNaturalist originals run to several thousand pixels
            # and a batch of them at native size exhausted the 9 GB free on a
            # shared GPU. A CUDA OOM also poisons the context, so every later
            # batch fails too -- which a bare `continue` turns into a silent
            # empty run. Give up instead of pretending to work.
            res = model.predict([str(p) for p in chunk], conf=a.conf, device=dev,
                                imgsz=a.imgsz, verbose=False)
            fails = 0
        except Exception as e:
            fails += 1
            print(f"    batch lỗi ({fails}): {str(e)[:90]}", flush=True)
            if fails >= 3:
                raise SystemExit("  ba lô liên tiếp hỏng -- dừng, không ghi kết quả giả")
            continue
        for p, r in zip(chunk, res):
            best, cls = 0.0, ""
            for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                if int(c) in ANIMAL and s > best:
                    best, cls = float(s), r.names[int(c)]
            fh.write(json.dumps({"file": str(p.relative_to(root)),
                                 "animal_conf": round(best, 4),
                                 "animal_cls": cls}) + "\n")
        fh.flush()
        if (i // a.batch) % 25 == 0:
            print(f"    {min(i+a.batch, len(todo))}/{len(todo)}", flush=True)
    fh.close()
    print("  xong")


if __name__ == "__main__":
    main()
