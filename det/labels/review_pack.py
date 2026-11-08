"""Review packages for the user: JPEG contact sheets with numbered, tier-coloured boxes
(A green, B yellow, ignore regions thin red) + review.csv to fill in. import_review()
applies the decisions to the label store."""
import csv, io, zipfile
from pathlib import Path
from PIL import Image, ImageDraw
from det.labels.store import load_rgb

COL = {"A": (0, 200, 0), "B": (255, 200, 0), "C": (255, 0, 0), "H": (0, 128, 255)}
CELL = (480, 300)
README = ("Fill the 'decision' column of review.csv:\n"
          "  ok          every bear has a correct box\n"
          "  bad:<ids>   these boxes have the wrong extent (ids are the numbers drawn on the boxes)\n"
          "  extra:<ids> these boxes are not on a bear\n"
          "  nobear      the image has no bear at all\n"
          "  missed      a bear has no box (fixed later in the HTML editor)\n"
          "Colours: green = tier A (two detectors agree), yellow = tier B (one detector),\n"
          "thin red = region ignored in training.\n")


def _cell(row, idx):
    img = load_rgb(row["path_hint"])
    s = min(CELL[0] / img.width, CELL[1] / img.height)
    im = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))))
    d = ImageDraw.Draw(im)
    for b in row["ignore"]:
        d.rectangle([v * s for v in b], outline=COL["C"], width=1)
    for k, b in enumerate(row["boxes"]):
        x1, y1, x2, y2 = [v * s for v in b["xyxy"]]
        d.rectangle([x1, y1, x2, y2], outline=COL.get(b.get("tier"), COL["H"]), width=3)
        d.rectangle([x1, y1, x1 + 14, y1 + 12], fill=(0, 0, 0))
        d.text((x1 + 3, y1), str(k), fill=(255, 255, 255))
    canvas = Image.new("RGB", (CELL[0], CELL[1] + 18), (30, 30, 30))
    canvas.paste(im, (0, 18))
    ImageDraw.Draw(canvas).text((4, 3), f"#{idx} {row['domain']} {Path(row['path_hint']).name[:58]}",
                                fill=(255, 255, 255))
    return canvas


def contact_sheets(rows, out_zip, per_sheet=12, cols=3):
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["sheet", "idx", "sha1", "path", "box_ids", "decision", "missed", "notes"])
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for si in range(0, len(rows), per_sheet):
            chunk = rows[si:si + per_sheet]
            nrow = (len(chunk) + cols - 1) // cols
            sheet = Image.new("RGB", (CELL[0] * cols, (CELL[1] + 18) * nrow), (0, 0, 0))
            for k, r in enumerate(chunk):
                idx = si + k
                sheet.paste(_cell(r, idx), ((k % cols) * CELL[0], (k // cols) * (CELL[1] + 18)))
                w.writerow([si // per_sheet, idx, r["sha1"], r["path_hint"],
                            " ".join(map(str, range(len(r["boxes"])))), "ok", "", ""])
            b = io.BytesIO()
            sheet.save(b, "JPEG", quality=85)
            z.writestr(f"sheet_{si // per_sheet:04d}.jpg", b.getvalue())
        z.writestr("review.csv", buf.getvalue())
        z.writestr("README.txt", README)


def import_review(csv_path, store, reviewer="user"):
    stats = {}
    for r in csv.DictReader(open(csv_path)):
        row = store.get(r["sha1"])
        if row is None:
            continue
        dec = (r["decision"] or "ok").strip()
        kind = dec.split(":")[0].strip()
        stats[kind] = stats.get(kind, 0) + 1
        ids = {int(i) for i in dec.split(":", 1)[1].replace(",", " ").split()} if ":" in dec else set()
        if kind == "ok":
            row["status"] = "reviewed"
        elif kind in ("bad", "extra"):
            row["ignore"] += [b["xyxy"] for i, b in enumerate(row["boxes"]) if i in ids]
            row["boxes"] = [b for i, b in enumerate(row["boxes"]) if i not in ids]
            row["status"] = "corrected" if row["boxes"] else "masked"
        elif kind == "nobear":
            row["status"] = "excluded"
        elif kind == "missed":
            row["status"] = "masked"
        else:
            raise ValueError(f"unknown decision {dec!r} for {r['sha1']}")
        row["n_det"] = len(row["boxes"])
        row["reviewer"] = reviewer
        store.put(row)
    store.save()
    return stats


# ---------------------------------------------------------------- HTML box editor
EDITOR_TEMPLATE = Path(__file__).with_name("editor_template.html")
EDITOR_MAX_SIDE = 1280


def editor_pack(rows, out_zip):
    """editor.html (single file, boxes inlined so it works from file://) + img/NNNN.jpg
    downscaled to <= 1280 px + boxes.json (same data, for scripts)."""
    import json
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    items = []
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for i, r in enumerate(rows):
            img = load_rgb(r["path_hint"])
            s = min(1.0, EDITOR_MAX_SIDE / max(img.size))
            if s < 1.0:
                img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR)
            b = io.BytesIO()
            img.save(b, "JPEG", quality=90)
            f = f"img/{i:04d}.jpg"
            z.writestr(f, b.getvalue())
            items.append({"sha1": r["sha1"], "file": f, "name": Path(r["path_hint"]).name, "scale": s,
                          "boxes": [[v * s for v in bx["xyxy"]] for bx in r["boxes"]]})
        js = json.dumps(items)
        z.writestr("editor.html", EDITOR_TEMPLATE.read_text().replace("/*BOXES*/[]", js))
        z.writestr("boxes.json", js)
        z.writestr("README.txt", "Giải nén, mở editor.html bằng Chrome/Firefox. Sửa khung, rồi bấm Export và gửi lại "
                                 "file boxes_reviewed.json.\n")


def import_editor(json_path, store, reviewer="user"):
    """Apply an exported boxes_reviewed.json (boxes already in original pixels)."""
    import json
    n = 0
    for it in json.load(open(json_path)):
        row = store.get(it["sha1"])
        if row is None:
            continue
        if it.get("nobear"):
            row["status"] = "excluded"
            row["boxes"] = []
        else:
            row["boxes"] = [{"xyxy": [float(v) for v in b], "score": 1.0, "tier": "H", "src": "human",
                             "truncated": bool(b[0] <= 2 or b[1] <= 2 or b[2] >= row["w"] - 2 or b[3] >= row["h"] - 2),
                             "crowd": False} for b in it["boxes"]]
            row["status"] = "reviewed"
        row["n_det"] = len(row["boxes"])
        row["reviewer"] = reviewer
        store.put(row)
        n += 1
    store.save()
    return n
