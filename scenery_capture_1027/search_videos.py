"""Pick scenery videos for the 27/10 rock/garden/bamboo negative capture (see
bear_project/docs/reports/2026-10-27_v19_fp/REPORT.md). No bears: titles mentioning
animals are skipped; frames are screened again later with YOLO."""
import json, yt_dlp
QUERIES = {
    "japanese_garden": ["Japanese garden 4K relaxing", "Japanese garden pond rocks 4K", "Kyoto garden 4K scenery"],
    "rock_garden": ["zen rock garden Kyoto 4K", "Ryoanji rock garden 4K", "karesansui dry garden 4K"],
    "bamboo": ["Arashiyama bamboo forest 4K", "bamboo forest walk 4K", "bamboo grove Japan 4K"],
    "rocks_stream": ["mountain stream boulders 4K nature", "Japan river rocks forest 4K", "moss covered rocks forest 4K"],
    "forest_trail": ["Japan forest trail hiking 4K", "Kumano Kodo 4K walk", "Yakushima forest 4K"],
    "scenery_tv": ["Japan 4K scenery relaxing film", "4K Japan nature cinematic", "Japan autumn garden 4K ambient"],
}
BAD = ("bear", "animal", "wildlife", "monkey", "deer", "boar", "cat", "dog", "zoo", "kuma", "熊", "クマ")
picked, seen = [], set()
with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True}) as y:
    for group, qs in QUERIES.items():
        n = 0
        for q in qs:
            for e in y.extract_info(f"ytsearch10:{q}", download=False)["entries"]:
                t, d, i = (e.get("title") or ""), e.get("duration") or 0, e.get("id")
                if i in seen or not (240 <= d <= 4800) or any(b in t.lower() for b in BAD):
                    continue
                seen.add(i); picked.append({"id": i, "group": group, "duration": d, "title": t}); n += 1
                if n % 2 == 0:            # two per query -> ~6 per group
                    break
        print(group, n)
json.dump(picked, open("scenery_capture_1027/videos.json", "w"), indent=1, ensure_ascii=False)
print(len(picked), "videos")
