"""Download each picked video (1080p video-only, YouTube cookies), keep 60 frames spread over
it (3%..97%), delete the video -> scenery_capture_1027/frames/<group>__<id>__<k>.jpg. Resumable."""
import json, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yt_dlp

OUT = Path("/data/nas07_new/PersonalData/phuocthien/scenery_capture_1027")
CK = "/data/nas07_new/PersonalData/phuocthien/bear_project/www.youtube.com_cookies.txt"
N = 60


def one(v):
    vid, g, dur = v["id"], v["group"], v["duration"]
    fdir = OUT / "frames"; fdir.mkdir(exist_ok=True)
    if len(list(fdir.glob(f"{g}__{vid}__*.jpg"))) >= N - 2:
        return N
    vdir = OUT / "videos"; vdir.mkdir(exist_ok=True)
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "cookiefile": CK,
            "format": "bv*[height<=1080][ext=mp4]/bv*[height<=1080]", "outtmpl": str(vdir / "%(id)s.%(ext)s"), "retries": 5}
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            y.download([f"https://www.youtube.com/watch?v={vid}"])
    except Exception as e:
        print(f"{vid}: download failed: {str(e)[:150]}", flush=True)
        return 0
    files = sorted(vdir.glob(f"{vid}.*"))
    if not files:
        return 0
    src = files[0]
    n = 0
    for k in range(N):
        t = dur * (0.03 + 0.94 * k / (N - 1))
        dst = fdir / f"{g}__{vid}__{k:02d}.jpg"
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1", "-q:v", "2", str(dst)])
        n += r.returncode == 0 and dst.exists()
    src.unlink(missing_ok=True)
    print(f"{g} {vid}: {n} frames", flush=True)
    return n


if __name__ == "__main__":
    vids = json.load(open(OUT / "videos.json"))
    with ThreadPoolExecutor(3) as ex:
        tot = sum(ex.map(one, vids))
    print("TOTAL_FRAMES", tot, flush=True)
