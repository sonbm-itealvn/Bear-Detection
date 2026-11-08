"""Stratified, human-reviewed eval set (spec §2): per split 600 bear + 200 non_bear.
Bears: 75 multi-bear images (>= 2 boxes, any domain) first, then per-domain quotas;
paste-augmented images never. Novel-scene images (no CLIP>=0.90 train neighbour) are
preferred. non_bear: up to 100 hard (classifier's highest-scoring) + random rest."""
import random

QUOTA = {"inat": 100, "rare": 75, "ir": 125, "youtube": 100, "small": 50, "screen": 75, "multi": 75}
N_NONBEAR, N_HARD = 200, 100


def select_eval(rows_by_split, novel, seed=0, nb_hard=None):
    rng = random.Random(seed)
    out = {}
    for sp, rows in rows_by_split.items():
        rows = [r for r in rows if r.get("status") != "excluded"]
        bears = [r for r in rows if r["cls"] == "bear" and r["domain"] != "paste"]
        rng.shuffle(bears)
        bears.sort(key=lambda r: r["path_hint"] not in novel)          # novel first, stable
        chosen, used = [], set()

        def take(pool, q, label):
            got = 0
            for r in pool:
                if got == q:
                    break
                if r["sha1"] not in used:
                    chosen.append(r["sha1"]); used.add(r["sha1"]); got += 1
            if got < q:
                print(f"  {sp}: {label} short by {q - got} ({got}/{q})")

        take([r for r in bears if len(r["boxes"]) >= 2], QUOTA["multi"], "multi")
        for dom, q in QUOTA.items():
            if dom != "multi":
                take([r for r in bears if r["domain"] == dom], q, dom)
        nb = [r for r in rows if r["cls"] == "non_bear"]
        rng.shuffle(nb)
        hard = [r for r in nb if nb_hard and r["path_hint"] in nb_hard][:N_HARD]
        hs = {r["sha1"] for r in hard}
        rest = [r for r in nb if r["sha1"] not in hs][:N_NONBEAR - len(hard)]
        out[sp] = {"bear": chosen, "non_bear": [r["sha1"] for r in hard + rest]}
    return out
