"""Scenery negatives from Wikimedia Commons (free licences): Japanese gardens, rock / zen
gardens, ponds with stones, mossy boulders, bamboo forests. Landscape images >= 1280 px wide,
fetched as 1920-px thumbnails -> scenery_capture_1027/commons/<group>__<pageid>.jpg, with
attribution in commons_meta.jsonl. Resumable."""
import json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

OUT = Path("/data/nas07_new/PersonalData/phuocthien/scenery_capture_1027")
API = "https://commons.wikimedia.org/w/api.php"
S = requests.Session(); S.headers["User-Agent"] = "bear-detector-research/1.0 (university lab; negative training images)"
QUERIES = {
    "japanese_garden": ["japanese garden rocks", "japanese garden pond stones", "kyoto garden", "japanese garden autumn", "japanese stroll garden"],
    "rock_garden": ["zen garden", "karesansui", "rock garden kyoto temple", "dry landscape garden gravel"],
    "bamboo": ["bamboo forest", "bamboo grove kyoto", "bamboo path"],
    "rocks_forest": ["mossy boulders forest", "boulders river japan", "moss covered rocks", "stone path garden japan", "forest trail japan"],
}
BAD = ("bear", "monkey", "deer", "boar", "dog", "cat", "bird", "heron", "koi", "insect", "people", "portrait", "map", "plan", "drawing", "painting", "diagram")


def search(q, limit=300):
    out, cont = [], {}
    while len(out) < limit:
        p = {"action": "query", "format": "json", "generator": "search", "gsrsearch": f"filetype:bitmap {q}", "gsrnamespace": 6,
             "gsrlimit": 50, "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iiurlwidth": 1920, **cont}
        r = None
        for a in range(6):                       # Commons rate-limits with a non-JSON page
            try:
                r = S.get(API, params=p, timeout=60).json(); break
            except Exception:
                time.sleep(10 * (a + 1))
        if r is None:
            print("  giving up on", q, flush=True); break
        for pg in r.get("query", {}).get("pages", {}).values():
            ii = (pg.get("imageinfo") or [{}])[0]
            out.append({"pageid": pg["pageid"], "title": pg["title"], "w": ii.get("width", 0), "h": ii.get("height", 0),
                        "thumb": ii.get("thumburl"), "page": ii.get("descriptionurl"),
                        "license": (ii.get("extmetadata", {}).get("LicenseShortName") or {}).get("value", ""),
                        "artist": (ii.get("extmetadata", {}).get("Artist") or {}).get("value", "")[:200]})
        if "continue" not in r:
            break
        cont = r["continue"]; time.sleep(2.0)
    return out


def main():
    (OUT / "commons").mkdir(parents=True, exist_ok=True)
    meta, seen = [], set()
    for g, qs in QUERIES.items():
        for q in qs:
            res = search(q)
            keep = [x for x in res if x["thumb"] and x["w"] >= 1280 and 1.2 <= x["w"] / max(x["h"], 1) <= 2.4
                    and not any(b in x["title"].lower() for b in BAD) and x["pageid"] not in seen]
            for x in keep:
                seen.add(x["pageid"]); x["group"] = g; x["query"] = q; meta.append(x)
            print(f"{g:16s} {q!r:40s} found {len(res):4d} kept {len(keep):4d}", flush=True)
            time.sleep(3)
    with open(OUT / "commons_meta.jsonl", "w") as f:
        for x in meta:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    def get(x):
        dst = OUT / "commons" / f"{x['group']}__{x['pageid']}.jpg"
        if dst.exists():
            return 1
        for a in range(3):
            try:
                r = S.get(x["thumb"], timeout=60)
                if r.status_code == 200 and r.content[:3] == b"\xff\xd8\xff":
                    dst.write_bytes(r.content); return 1
                if r.status_code == 429:
                    time.sleep(5 * (a + 1))
            except Exception:
                time.sleep(2)
        return 0
    with ThreadPoolExecutor(2) as ex:
        n = sum(ex.map(get, meta))
    print("DOWNLOADED", n, "of", len(meta))


if __name__ == "__main__":
    main()
