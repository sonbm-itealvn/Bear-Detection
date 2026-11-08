"""Pick pet / animal / people videos for the 28/10 V19 false positives (close-up crows, cats,
a cat standing upright, a black dog lying down, a dark-clad man in a forest; see
bear_project/docs/reports/2026-10-27_v19_fp/REPORT.md section 5). Titles naming a bear are skipped;
frames are checked again later."""
import json, yt_dlp
OUT = "/data/nas07_new/PersonalData/phuocthien/animals_capture_1029"
QUERIES = {
    "crow": ["crows close up Japan", "カラス 近く 4K", "crow eating on the ground close up", "large-billed crow Japan", "crows in the city parking lot"],
    "cat_closeup": ["stray cats Japan eating", "black cat close up", "cat standing on hind legs", "野良猫 4K", "cats in the grass playing", "cat island Japan cats"],
    "dog_dark": ["black German shepherd", "black dog lying down close up", "black labrador relaxing", "black shiba inu", "kai ken dog"],
    "person_forest": ["hikers walking forest trail", "man walking in the woods", "forest walk people Japan", "hiking in the forest vlog", "mushroom picking in the forest"],
}
BAD = ("bear", "kuma", "熊", "クマ", "cartoon", "animation", "anime", "minecraft", "game", "drawing", "#shorts")
picked, seen = [], set()
with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True}) as y:
    for group, qs in QUERIES.items():
        n = 0
        for q in qs:
            k = 0
            for e in y.extract_info(f"ytsearch12:{q}", download=False)["entries"]:
                t, d, i = (e.get("title") or ""), e.get("duration") or 0, e.get("id")
                if i in seen or not (150 <= d <= 4800) or any(b in t.lower() for b in BAD):
                    continue
                seen.add(i); picked.append({"id": i, "group": group, "duration": d, "title": t}); n += 1; k += 1
                if k == 2:
                    break
        print(group, n, flush=True)
json.dump(picked, open(f"{OUT}/videos.json", "w"), indent=1, ensure_ascii=False)
print(len(picked), "videos")
