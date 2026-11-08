"""Camera acceptance scoring, v16 / v17 / v18 sessions.
Truth comes from rename_map.csv `true_class`, not from the session's own `truth` (derived from the directory).
See acceptance_full.py for the v18 report.

    python acceptance_report_1016.py"""
import csv, json, os, re
from collections import defaultdict
from pathlib import Path

BASE = Path("/data/nas07_new/PersonalData/phuocthien")
MAN = {"video": BASE / "video_categorized_grouped_1in6/rename_map.csv",
       "images": BASE / "acceptance_merged/rename_map.csv"}
SESSIONS = [  # (label, benchmark, responses)
    ("v16 18/09", "images", BASE / "bear_project/acc0918/images/_acceptance_session/responses.jsonl"),
    ("v18 16/10", "images", BASE / "acceptance_1016/img/_acceptance_session/responses.jsonl"),
    ("v16 18/09", "video", BASE / "bear_project/acc0918/video/_acceptance_session/responses.jsonl"),
    ("v17 02/10", "video", BASE / "acc_1002/_acceptance_session/responses.jsonl"),
    ("v18 16/10", "video", BASE / "acceptance_1016/vid/_acceptance_session/responses.jsonl"),
]
ACC = re.compile(r"^acc\d{4}_(?:(images|video)__)?(bear|non_bear)__(.+?)__MFDC")


def burned_keys():
    """(benchmark, class, stem) of every reference with camera frames in train."""
    keys = set()
    for cls in ("bear", "non_bear"):
        for f in os.listdir(BASE / "bear_dataset_v_final_final/train" / cls):
            m = ACC.match(f)
            if m:
                keys.add((m.group(1) or "video", m.group(2), m.group(3)))
    return keys


def replay(path):
    ans = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("event") == "undo":
                ans.pop(r.get("file"), None)
            elif "answer" in r and r.get("file"):
                ans[r["file"]] = r["answer"]
    return ans


def stats(c):
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    rec = tp / (tp + fn) if tp + fn else float("nan")
    pre = tp / (tp + fp) if tp + fp else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return rec, pre, spec


def main():
    burned = burned_keys()
    print(f"burned references (camera frames in train): {len(burned)}")
    man = {k: {r["new_path"]: r for r in csv.DictReader(open(p, newline="", encoding="utf-8"))} for k, p in MAN.items()}
    rows = []
    for label, bench, resp in SESSIONS:
        if not resp.exists():
            print(f"  missing {resp}")
            continue
        ans = replay(resp)
        cells = defaultdict(lambda: dict(tp=0, fn=0, fp=0, tn=0))
        for p, v in ans.items():
            r = man[bench].get(p)
            if r is None:
                continue
            truth = r["true_class"]
            key = (bench, p.split("/")[1], Path(p).stem)
            part = "burned" if key in burned else "fresh"
            hit = v in ("bear", "skip")
            for k in ("all", part, f"grp_{r['group']}"):
                c = cells[k]
                if truth == "bear":
                    c["tp" if hit else "fn"] += 1
                else:
                    c["fp" if hit else "tn"] += 1
        rows.append((label, bench, cells))
    print(f"\n{'session':<11}{'bench':<8}{'part':<8}{'n':>6}{'TP':>6}{'FN':>5}{'FP':>5}{'TN':>6}"
          f"{'recall':>9}{'prec':>8}{'spec':>8}")
    for label, bench, cells in rows:
        for part in ("all", "fresh", "burned"):
            c = cells[part]
            n = sum(c.values())
            rec, pre, spec = stats(c)
            print(f"{label:<11}{bench:<8}{part:<8}{n:>6}{c['tp']:>6}{c['fn']:>5}{c['fp']:>5}{c['tn']:>6}"
                  f"{100*rec:>8.1f}%{100*pre:>7.1f}%{100*spec:>7.1f}%")
        print()
    json.dump({f"{l}|{b}": {k: v for k, v in c.items()} for l, b, c in rows},
              open(BASE / "acceptance_1016/report_1016.json", "w"), indent=1)


if __name__ == "__main__":
    main()
