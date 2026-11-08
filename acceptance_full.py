"""Full v18 camera-acceptance report. Truth = rename_map.csv true_class. Last answer wins; undo erases.
Writes acceptance_1016/v18_acceptance_full.md and false_cases_v18.csv."""
import csv, json
from collections import defaultdict, Counter
from pathlib import Path

BASE = Path("/data/nas07_new/PersonalData/phuocthien")
RUNS = {"images": (BASE / "acceptance_merged/rename_map.csv", BASE / "acceptance_1016/img/_acceptance_session/responses.jsonl", "title"),
        "video": (BASE / "video_categorized_grouped_1in6/rename_map.csv", BASE / "acceptance_1016/vid/_acceptance_session/responses.jsonl", "subj_class")}


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


def fmt(c):
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    f = lambda a, b: f"{100 * a / b:.1f}%" if b else "—"
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0
    return [str(tp + fn + fp + tn), str(tp), str(fn), str(fp), str(tn), f(tp, tp + fn), f(tp, tp + fp), f(tn, tn + fp),
            f"{100 * f1:.1f}%" if tp + fp + fn else "—", f(tp + tn, tp + fn + fp + tn)]


HDR = "| {} | n | TP | FN | FP | TN | recall | precision | specificity | F1 | accuracy |\n|---|---|---|---|---|---|---|---|---|---|---|"


def main():
    L = ["# V19 ep58: kết quả nghiệm thu trên camera (15–16/10/2026)", "",
         "Build trên camera: `deliver_ptq_20261014_V19_regnetx_ep58` (PTQ, md5 `a51174d26206be3fab007413fcacb63f`).",
         "Nhãn đúng lấy từ cột `true_class` của `rename_map.csv`, không lấy từ tên thư mục. "
         "Một lần **skip được tính là đúng** cho cả hai lớp. Nếu một ảnh được trả lời nhiều lần thì lấy câu trả lời cuối; undo xóa câu trả lời.", ""]
    false_rows, summary = [], {}
    for bench, (manp, resp, sp_key) in RUNS.items():
        man = {r["new_path"]: r for r in csv.DictReader(open(manp, newline="", encoding="utf-8"))}
        ans = replay(resp)
        cells = defaultdict(lambda: defaultdict(lambda: dict(tp=0, fn=0, fp=0, tn=0)))
        n_skip = Counter()
        for p, v in ans.items():
            r = man[p]
            truth = r["true_class"]
            if v == "skip":
                n_skip[truth] += 1
                ok = True
            else:
                ok = (v == truth)
            k = ("tp" if ok else "fn") if truth == "bear" else ("tn" if ok else "fp")
            for dim, val in (("all", "all"), ("group", r["group"]), ("distance", r.get("distance_band") or "?"),
                             ("species", (r.get(sp_key) or "?") if truth == "bear" else None)):
                if val is not None:
                    cells[dim][val][k] += 1
            if not ok:
                false_rows.append([bench, "FN" if truth == "bear" else "FP", r["group"], p, truth, v,
                                   r.get("distance_band", ""), r.get(sp_key, ""), r.get("video", r.get("source", ""))])
        summary[bench] = cells["all"]["all"]
        L += [f"## {'Ảnh tĩnh' if bench == 'images' else 'Video'} ({len(ans)} ảnh được trả lời, skip: "
              f"{sum(n_skip.values())} ({dict(n_skip)}))", "", HDR.format(""), "| **tổng** | " + " | ".join(fmt(cells['all']['all'])) + " |", ""]
        for dim, title in (("group", "nhóm"), ("distance", "khoảng cách")):
            L += [f"**Theo {title}**", "", HDR.format(title)]
            for k in sorted(cells[dim]):
                L.append(f"| {k} | " + " | ".join(fmt(cells[dim][k])) + " |")
            L.append("")
        wrong = [fr for fr in false_rows if fr[0] == bench]
        if bench == "images":
            for kind, title in (("FN", "Ảnh gấu bị bỏ sót (FN)"), ("FP", "Ảnh không gấu bị báo là gấu (FP)")):
                rows_ = sorted((fr for fr in wrong if fr[1] == kind), key=lambda fr: (fr[2], fr[3]))
                L += [f"**{title}: {len(rows_)}**", "", "| nhóm | file | khoảng cách | mô tả |", "|---|---|---|---|"]
                L += [f"| {fr[2]} | `{fr[3]}` | {fr[6] or '—'} | {fr[7] or '—'} |" for fr in rows_]
                L.append("")
        else:
            per = defaultdict(lambda: Counter())
            for fr in wrong:
                per[fr[8]][fr[1]] += 1
            tot = Counter((man[p]["video"], man[p]["true_class"]) for p in ans)
            L += ["**Frame sai theo video nguồn**", "",
                  "| video | FN | frame gấu | FP | frame không gấu |", "|---|---|---|---|---|"]
            for vid in sorted(per, key=lambda v: -(per[v]["FN"] + per[v]["FP"])):
                L.append(f"| `{vid}` | {per[vid]['FN']} | {tot[(vid, 'bear')]} | {per[vid]['FP']} | {tot[(vid, 'non_bear')]} |")
            L.append("")
    L += ["## So sánh với các bản trước (cùng bộ benchmark)", "",
          "| model | ảnh tĩnh: recall / precision | video: recall / precision |", "|---|---|---|",
          "| V16 (18/09) | 95.9 / 93.0 | 85.4 / 90.6 |", "| V17 (02/10, build trừ mean ngược) | — | 66.1 / 97.3 |"]
    s = {b: fmt(c) for b, c in summary.items()}
    L += [f"| **V19 (16/10)** | **{s['images'][5]} / {s['images'][6]}** | **{s['video'][5]} / {s['video'][6]}** |", "",
          "Số của V16 và V17 tính theo quy ước cũ: skip tính là đã phát hiện gấu.", "",
          f"Danh sách từng ảnh sai: `acceptance_1016/false_cases_v18.csv` ({len(false_rows)} dòng)."]
    (BASE / "acceptance_1016/v18_acceptance_full.md").write_text("\n".join(L) + "\n")
    with open(BASE / "acceptance_1016/false_cases_v18.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "kind", "group", "file", "true_class", "device_said", "distance_band", "species_or_subject", "video_or_source"])
        w.writerows(false_rows)
    print("\n".join(L))


if __name__ == "__main__":
    main()
