"""One offline review page per package (replaces contact sheets + review.csv + editor):
every image is shown with its auto boxes; Space = correct, x = no bear, edit boxes in place.
Export writes ket_qua_<title>.json; import_app() applies it to the label store."""
import io, json, zipfile
from pathlib import Path

from PIL import Image

from det.labels.store import load_rgb

TEMPLATE = Path(__file__).with_name("review_app.html")
GUIDE = """HƯỚNG DẪN
1. Giải nén thư mục này, mở review.html bằng Chrome.
2. Mỗi ảnh: khung đúng hết -> Space. Khung sai -> bấm vào khung rồi Delete, hoặc kéo góc để sửa.
   Gấu thiếu khung -> kéo chuột vẽ khung. Không có gấu -> x.
   Ảnh ghi "KHÔNG GẤU": đúng là không có gấu -> Space; nếu thật ra có gấu -> vẽ khung rồi Space.
3. Khung đỏ mảnh = vùng bỏ qua, không phải nhãn (phím h để ẩn). Nếu trong khung đỏ có con gấu chưa có
   khung xanh/vàng: double-click vào khung đỏ để biến nó thành khung gấu, rồi kéo góc chỉnh cho vừa.
4. Tiến độ tự lưu trong trình duyệt. Xong thì bấm "Export kết quả" và gửi file ket_qua_*.json.
Phím: Space đúng | x không có gấu | Delete xoá khung | r khôi phục | n/p sau/trước | u ảnh chưa duyệt
"""


def app_pack(rows, out_zip, title, max_side=800, quality=78):
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    items = []
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for i, r in enumerate(rows):
            img = load_rgb(r["path_hint"])
            s = min(1.0, max_side / max(img.size))
            if s < 1.0:
                img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR)
            b = io.BytesIO()
            img.save(b, "JPEG", quality=quality)
            f = f"img/{i:04d}.jpg"
            z.writestr(f, b.getvalue())
            items.append({"sha1": r["sha1"], "file": f, "name": Path(r["path_hint"]).name, "scale": s,
                          "cls": r["cls"], "domain": r.get("domain", ""),
                          "boxes": [[v * s for v in bx["xyxy"]] + [bx.get("tier", "A")] for bx in r["boxes"]],
                          "ignore": [[v * s for v in g] for g in r.get("ignore", [])]})
        pack = json.dumps({"title": title, "items": items}, indent=0)
        z.writestr("review.html", TEMPLATE.read_text().replace('/*PACK*/{"title": "", "items": []}', pack))
        z.writestr("HUONG_DAN.txt", GUIDE)
    return len(items)


def import_app(json_path, store, reviewer="user"):
    stats = {"ok": 0, "edited": 0, "nobear": 0, "nonbear_has_bear": 0, "unseen": 0}
    for it in json.load(open(json_path)):
        row = store.get(it["sha1"])
        if row is None:
            continue
        dec = it.get("decision")
        if not dec:
            stats["unseen"] += 1
            continue
        boxes = [{"xyxy": [float(v) for v in b[:4]], "tier": b[4] if len(b) > 4 else "H",
                  "src": "human" if (len(b) <= 4 or b[4] == "H") else "auto+confirmed", "score": 1.0,
                  "truncated": bool(b[0] <= 2 or b[1] <= 2 or b[2] >= (row["w"] or 1e9) - 2 or b[3] >= (row["h"] or 1e9) - 2),
                  "crowd": False} for b in it.get("boxes", [])]
        if row["cls"] != "bear":
            if boxes:                         # the negative actually shows a bear: flag it, never move files
                row.update(status="excluded", has_bear=True, boxes=boxes)
                stats["nonbear_has_bear"] += 1
            else:
                row["status"] = "reviewed"
                stats["ok"] += 1
        elif dec == "nobear":
            row.update(status="excluded", boxes=[])
            stats["nobear"] += 1
        else:
            if dec == "edited":
                row["boxes"] = boxes
                stats["edited"] += 1
            else:
                stats["ok"] += 1
            row["status"] = "reviewed" if row["boxes"] else "masked"
        row["n_det"] = len(row["boxes"])
        row["reviewer"] = reviewer
        row["review_decision"] = dec
        store.put(row)
    store.save()
    return stats
