"""
collect_confusable_animals.py

Collect elk, raccoon, moose, and other bear-confusable animals
from iNaturalist API and LILA BC datasets for non_bear training data.

These animals are FP root causes:
- Elk/Moose: dark coat + bulky silhouette similar to bear
- Raccoon: rounded dark body, trail camera common
- Wild boar: dark fur, stocky build
- Dark dogs/coyotes: common in trail cam footage

Usage:
    # Download from iNaturalist (no API key needed, public API)
    python collect_confusable_animals.py --source inat --output non_bear_confusable --n 500

    # Download from LILA BC (requires dataset URLs)
    python collect_confusable_animals.py --source lila --output non_bear_confusable --n 500

    # Both sources
    python collect_confusable_animals.py --source both --output non_bear_confusable --n 1000

    # With CLIP filtering to keep only trail-cam-like images
    python collect_confusable_animals.py --source inat --output non_bear_confusable --n 500 --clip-filter

    # With Gemini verification (ensure no bear in images)
    python collect_confusable_animals.py --source inat --output non_bear_confusable --n 500 --gemini-verify
"""

import os
import sys
import time
import json
import random
import shutil
import hashlib
import argparse
import requests
import subprocess
import pickle
import threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode

# ============================================================
# TARGET SPECIES — iNaturalist taxon IDs
# ============================================================
# WARNING: this first INAT_TAXA is DEAD CODE. The "# Corrected taxon IDs" dict
# below rebinds the name and replaces it entirely, so anything added here is
# silently ignored -- the collector then prints "Unknown species" and moves on.
# That cost a full collection run on 2026-09-24 (13 of 14 species skipped).
# Add new taxa to the dict under "# Corrected taxon IDs".
INAT_TAXA = {
    # Primary FP causes
    "elk":         41782,   # Cervus canadensis
    "moose":       41774,   # Alces alces
    "raccoon":     41680,   # Procyon lotor
    "wild_boar":   42134,   # Sus scrofa
    # Secondary FP causes
    "coyote":      41540,   # Canis latrans
    "wolf":        41543,   # Canis lupus
    "deer":        41780,   # Odocoileus (whitetail/mule)
    "bison":       41770,   # Bison bison
    "mountain_lion": 41680, # Puma concolor (placeholder, use correct ID below)
}

# Corrected taxon IDs
INAT_TAXA = {
    # Genus-widening for the rare Japanese dark-furred species, 2026-09-24.
    # Japanese marten has only 160 research-grade observations and 120 of them
    # are the held-out test set, leaving 40 to train on -- for one of the worst
    # false-positive species (20%). For a NEGATIVE, looking alike is what
    # matters, not the exact species; sable and mountain hare also occur in
    # Hokkaido.
    "sable":                 41795,   # Martes zibellina (178 obs)
    "pine_marten":           41796,   # Martes martes
    "stone_marten":          41800,   # Martes foina
    "yellow_throated_marten": 41791,  # Martes flavigula
    "mountain_hare":         55908,   # Lepus timidus
    "european_hare":         43128,   # Lepus europaeus
    "asian_badger":          74558,   # Meles leucurus
    # --- Dark-furred small/medium mammals, added 2026-09-24 -----------------
    # The customer found V16 calling dark-furred animals from web photos bears.
    # Measured on a held-out daylight set (darkfur_test_2409, 14 species x 120):
    # V16 5.7% false positives, the current run 9.1%, worst on mustelids,
    # squirrels, civet and cat -- NOT foxes or rabbits, which sat at 1-6%.
    # Daylight coverage in train was 221 foxes and ZERO rabbits; the rest of
    # the 2,831 foxes / 222 rabbits are night trail-cam frames.
    "gray_fox":              42076,   # Urocyon cinereoargenteus
    "japanese_hare":         43139,   # Lepus brachyurus (193 research obs)
    "european_rabbit":       43151,   # Oryctolagus cuniculus, incl. dark domestic
    "eastern_cottontail":    43111,   # Sylvilagus floridanus
    "japanese_marten":       41794,   # Martes melampus (160 research obs)
    "japanese_badger":       74557,   # Meles anakuma -- worst species for V16, 20.8%
    "masked_palm_civet":     41625,   # Paguma larvata, invasive in Japan
    "domestic_cat":         118552,   # Felis catus
    "raccoon_dog_common":   855311,   # Nyctereutes procyonoides
    "japanese_squirrel":     46000,   # Sciurus lis
    "eurasian_red_squirrel": 46001,   # Sciurus vulgaris, incl. dark morph
    "american_mink":       1264432,   # Neogale vison
    "eurasian_otter":        41850,   # Lutra lutra
    # Relatives of the worst held-out species, added 2026-09-26: Japanese
    # badger and marten are near their iNat ceiling (598 / 163 research obs),
    # so the top-up widens to the same body plan instead.
    "american_badger":       41789,   # Taxidea taxus
    "na_river_otter":        41777,   # Lontra canadensis
    "asian_smallclawed_otter": 1638835,  # Aonyx (Lutra) cinerea

    "elk":               41782,
    "moose":             41774,
    "raccoon":           41680,
    "wild_boar":         42134,
    "coyote":            41540,
    "wolf":              41543,
    "white_tailed_deer": 42159,
    "mule_deer":         42158,
    "bison":             41770,
    "mountain_lion":     41545,
    "elephant":          43694,  # Loxodonta africana (African bush elephant) — most observations on iNat
    "asian_elephant":    42863,  # Elephas maximus (Asian elephant)
    # FP confusables observed in partner screen-recording FP videos (2026-06):
    # boar already above; these add the fox / monkey / dog / fluffy-cat cases.
    "red_fox":           42069,  # Vulpes vulpes (incl. silver/black colour morph, on snow)
    "japanese_macaque":  43458,  # Macaca fuscata (dark "snow monkeys" on snow)
    "sika_deer":         42210,  # Cervus nippon -- ground truth for the new deer
                                  # class; inat_verify_cache.pkl has NO deer at
                                  # all (elk/moose are the wrong species), so
                                  # this taxon had never been collected before.
    "domestic_dog":      47144,  # Canis familiaris (dark breeds — verify-heavy, noisy source)
    "pallas_cat":        42029,  # Otocolobus manul (dense fluffy fur; few obs ~237)
    "bobcat":            41976,  # Lynx rufus (fluffy wild cat, high volume substitute)
    # Bear in urban/industrial context — same taxon as bear but filtered by place
    "bear_urban":        41688,  # Ursus americanus, filtered by urban tags
    # FP confusable found in acceptance ir_neg video 98MkQYto4eA (2026-07):
    # European Badger emerging from a den/sett at night -- dark body, IR
    # night-vision, burrow context -- dominant FP source in that video
    # (~150+/186 FP frames are badger dens/badgers).
    "european_badger":   855297,  # Meles meles

    # --- BEAR SPECIES, added 2026-09-18. These are POSITIVES, not confusables:
    # collect them into a separate --output tree and merge into the bear class.
    # Motivation: the 18/09 acceptance session missed 89 video frames, 59 of them
    # in one BBC documentary, and the contact sheet shows the misses are
    # overwhelmingly species the training set has ZERO labelled examples of --
    # `train/bear` contains no file named panda, polar, kermode, spirit or
    # sun_bear, and is 31k frames of North American black/brown bear.
    # Observation counts are research-grade, checked against the API on the day.
    "asian_black_bear":  41647,   #   360  Ursus thibetanus -- THE Japanese bear
    "brown_bear":        41641,   # 17135  Ursus arctos -- Hokkaido
    "polar_bear":        41644,   #  2961  Ursus maritimus
    "sun_bear":          41655,   #   482  Helarctos malayanus
    "sloth_bear":        41651,   #   867  Melursus ursinus
    "spectacled_bear":   41657,   #  1090  Tremarctos ornatus
    "kermode_bear":     123749,   #    54  U. a. kermodei -- the cream "spirit bear"
    "giant_panda":       41659,   #    28  Ailuropoda melanoleuca -- see note below
    # Giant panda is the single largest miss category in the video benchmark, and
    # iNaturalist has only 28 research-grade observations of it (pandas live in
    # reserves, so wild records are rare). iNat cannot fix this one; it needs
    # YouTube via collect_bear_sahi.py or another source.
    # FP confusables found in acceptance day_neg/ir_neg "wildlife compilation"
    # videos (2026-07): _qlckdZwy2M (bull at market), hDo0aWkEsmU (lions on
    # patio). Compilation videos mix many unrelated species into what were
    # assumed to be pure trail-cam negatives.
    "lion":              41964,  # Panthera leo
    "water_buffalo":     81925,  # Bubalus bubalis (dark quadruped, market/village context)
    "domestic_cattle":   74113,  # Bos taurus (dark breeds; alternate ID for the same FP frame type)
    # FP confusable found via Novaic 1-in-6 acceptance run false-frame review
    # (2026-08-18): leopard triggered a false positive in a news-clip
    # compilation negative video ("Residents find leopard hiding in
    # neighbour's courtyard" -- text-chyron broadcast content, same pattern as
    # the lion/cattle FPs above). Not previously in this taxon list.
    "leopard":           41963,  # Panthera pardus (14107 obs, verified via live API 2026-08-18)
    # "monkey" class scope widened 2026-09-07 (user decision): train on the
    # whole Macaca genus, not just M. fuscata -- japanese_macaque alone has
    # only 2,902 iNat observations, far short of collection targets, and
    # NTLNP (the other major species-expansion source this session) has no
    # monkey data at all. Species below chosen for morphological closeness to
    # M. fuscata (short/no tail, robust cold-adapted body) -- explicitly
    # EXCLUDES M. fascicularis (long-tailed macaque, 24,713 obs, by far the
    # most abundant on iNat) and M. radiata (bonnet macaque, distinctive head
    # hair whorl) because their silhouette differs too much from the
    # deployment-target species; a same-genus swap is the same tradeoff
    # already accepted for boar/deer via NTLNP (different geography, same
    # species) but pushed one level further (same genus, different species),
    # so still route every image through the existing species-verify Bedrock
    # pipeline before merge -- do not merge on iNat's taxon label alone.
    # Counts verified live via iNaturalist API 2026-09-07.
    "formosan_rock_macaque": 43450,  # Macaca cyclopis (6,821 obs)
    "rhesus_macaque":        43460,  # Macaca mulatta (10,714 obs)
    "tibetan_macaque":       43451,  # Macaca thibetana (740 obs)
    "barbary_macaque":       43452,  # Macaca sylvanus (3,272 obs, tailless)
    # "deer" class scope widened 2026-09-07 (user decision), same reasoning as
    # the monkey genus widening above: sika_deer (Cervus nippon) alone has a
    # hard ceiling of only 9,764 iNat observations, not enough to reach
    # collection targets even at 100% yield. Unlike the macaque case this
    # crosses genus boundaries (Cervidae family, not genus Cervus) -- chosen
    # for the same white-spotted-coat + small/medium body silhouette as sika,
    # explicitly NOT elk/red-deer-sized species (those are tracked separately
    # as bear-confusable non_bear negatives elsewhere in INAT_TAXA -- keep
    # them there, do not let this list bleed into that one). Route through
    # the same Bedrock species-verify pipeline before merge; do not trust the
    # iNat taxon label alone given the cross-genus jump.
    # Counts verified live via iNaturalist API 2026-09-07.
    "fallow_deer": 42161,  # Dama dama (19,767 obs, white-spotted coat)
    "chital":      42166,  # Axis axis (9,843 obs, white-spotted coat)
    # POSITIVE-class additions 2026-09-09 (user decision, overriding this
    # session's own earlier exclusions): these three were originally proposed
    # as hard NEGATIVES for boar/deer/monkey (too visually different from the
    # target species, per the reasoning above) but the user decided they
    # should count as POSITIVE examples for their respective class instead.
    # Route to boar/deer/monkey at merge time, same as every other species in
    # this genus/family-widened set -- still go through the local+Bedrock
    # verify pipeline, do not trust the taxon label alone.
    "red_deer":            204113,  # Cervus elaphus (21,095 research-grade) -> deer
                                     # NOTE: notably larger than sika/fallow/chital;
                                     # this reverses the "reject as OTHER" rule in
                                     # reverify_deer_boar_strict.py / verify_ntlnp_
                                     # bedrock.py / verify_species_mismatch_bedrock.py
                                     # -- those prompts must be updated (or a new
                                     # pass run) before this can be trusted as DEER
                                     # by the existing verify scripts.
    "long_tailed_macaque": 43459,   # Macaca fascicularis (22,292 research-grade) -> monkey
                                     # NOTE: this is the exact species the monkey
                                     # verify prompt currently instructs Bedrock to
                                     # REJECT as OTHER ("long tail, visibly longer
                                     # than the body") -- same prompt-update caveat.
    "peccary":             42113,   # Pecari tajacu (18,665 research-grade) -> boar
                                     # NOTE: family Tayassuidae, NOT Suidae -- not
                                     # actually the same family as wild boar, just
                                     # superficially pig-shaped and native to a
                                     # different continent. Biologically a bigger
                                     # stretch than the same-genus/same-family swaps
                                     # used elsewhere in this list; flagging for the
                                     # record, proceeding per explicit user decision.

    # HARD-NEGATIVE additions 2026-09-09 (non_bear, NOT any positive class):
    # proposed after the boar/deer/monkey genus/family widening above created
    # three concrete, previously-undocumented confusable gaps -- the model
    # currently has zero real training images of these near-miss species and
    # only learns to reject them indirectly, through the Bedrock verify
    # prompts' rejection lists, at collection time.
    "domestic_pig":        324492,  # Sus scrofa domesticus (2,308 obs) -- same
                                     # species as wild_boar, farm/captive context
    "japanese_raccoon_dog": 855310, # Nyctereutes viverrinus (1,572 obs) -- tanuki,
                                     # Japan-endemic, common IR trail-cam confusable
    "japanese_serow":      74140,   # Capricornis crispus (480 obs) -- Japan-endemic,
                                     # co-occurs with boar/sika deer in Japan camera
                                     # traps (Fukushima Daiichi monitoring dataset)
    "domestic_goat":       123070,  # Capra hircus (8,806 obs)
    "domestic_sheep":      121578,  # Ovis aries (4,262 obs)
    # Gorilla added 2026-09-09 (user decision) -- NOT a monkey hard-negative
    # (too visually different from macaque -- no tail, knuckle-walking,
    # much larger -- would not plausibly be confused with the monkey class).
    # Routed to non_bear instead: large body + dark fur matches this
    # project's own documented bear false-positive pattern ("model
    # over-triggers on large dark masses" -- CLAUDE.md Pending Tasks),
    # same rationale as the existing lion/elephant/water_buffalo entries.
    "western_gorilla": 43580,  # Gorilla gorilla (400 research-grade)
    "eastern_gorilla": 68790,  # Gorilla beringei (1,252 research-grade)
}

# ============================================================
# TEXT-FILTERED TAXA — iNat has no dedicated taxon for these (breed / color
# morph), so we constrain by taxon_id + free-text search on tags/description
# instead. Query volumes verified against the live API (2026-07-13):
#   dog "black"=369, "schnauzer"=78, "black lab"=39, "newfoundland"=12,
#       "scottish terrier"=8, "black terrier"=6, "black puppy"=7 (quality=any)
#   fox "black fox"=168, "dark fox"=76, "melanistic"=54, "melanistic fox"=36,
#       "black morph"=17 (quality=any). NOTE: "silver fox" is NOT usable —
#       it matches ~116k obs (nearly the whole species), evidently token-OR
#       matching rather than phrase matching.
# ============================================================
INAT_TEXT_QUERIES = {
    # Compact fluffy black dog breeds (Scottish Terrier-like) — the confirmed
    # FP driver in MFDC0556.MP4 (25/25 frames, p_bear=0.997 mean).
    "dark_dog": (47144, ["black", "schnauzer", "black lab", "scottish terrier",
                         "newfoundland", "black terrier", "black puppy"]),
    # Melanistic/dark color-morph red fox — the confirmed FP driver in
    # MFDC0539.MP4 (25/25 frames, p_bear=0.987 mean).
    "dark_fox": (42069, ["black fox", "dark fox", "melanistic", "melanistic fox",
                         "black morph"]),
}

# LILA BC dataset info — Snapshot Serengeti style datasets have labeled animals
LILA_DATASETS = {
    "idaho_camera_traps": {
        "url": "https://lila.science/datasets/idaho-camera-traps/",
        "species": ["elk", "moose", "deer", "coyote", "raccoon"],
        "metadata_url": "https://storage.googleapis.com/public-datasets-lila/idaho-camera-traps/idaho_camera_traps_metadata.json.zip",
    },
    "caltech_camera_traps": {
        "url": "https://lila.science/datasets/caltech-camera-traps/",
        "species": ["coyote", "raccoon", "deer"],
        "metadata_url": "https://storage.googleapis.com/public-datasets-lila/caltech-camera-traps/CaltechCameraTrapsECCV18.json.zip",
    },
}


# ============================================================
# iNaturalist COLLECTOR
# ============================================================
class INatCollector:
    BASE_URL = "https://api.inaturalist.org/v1"

    def __init__(self, output_dir, n_per_species=200, quality="research",
                 exclude_obs=None):
        # Observation ids to skip: ones already in train or in the held-out
        # test set (a top-up must neither re-download nor leak into test).
        self.exclude_obs = exclude_obs or set()
        self.output_dir = Path(output_dir)
        self.n_per_species = n_per_species
        self.quality = quality  # "research" = verified ID
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "BearDetectionDataCollector/1.0"

    def search_observations(self, taxon_id, page=1, per_page=200, extra_params=None):
        """Search iNaturalist observations with photos."""
        params = {
            "taxon_id": taxon_id,
            "quality_grade": self.quality,
            "has[]": "photos",
            "per_page": min(per_page, 200),
            "page": page,
            # id/asc, NOT random: random+page pagination on iNat's API is
            # unstable at depth (confirmed 2026-09-09 -- a wild_boar pull
            # requesting --n 18000 against a 49,265 research-grade ceiling
            # silently stopped at 9,079, well short of both, because deep
            # pages under order=random intermittently return empty/duplicate
            # results). id/asc paginates deterministically through the whole
            # result set; since collection now targets exhausting each
            # species' real ceiling rather than a small random subsample,
            # deterministic order introduces no meaningful bias.
            "order": "asc",
            "order_by": "id",
            "photos": "true",
        }
        if extra_params:
            params.update(extra_params)
        # Retry: the caller treats None as "no more results" and ends the
        # species, so one timeout used to end a pull at 0 (european_rabbit,
        # 26/09, against 49k research-grade observations).
        for attempt in range(5):
            try:
                r = self.session.get(f"{self.BASE_URL}/observations", params=params, timeout=30)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                print(f"    iNat API error (attempt {attempt + 1}/5): {e}")
                time.sleep(5 * 2 ** attempt)
        return None

    def get_photo_urls(self, observation):
        """Extract photo URLs from observation."""
        urls = []
        for photo in observation.get("photos", []):
            url = photo.get("url", "")
            if url:
                url = url.replace("square", "medium").replace("small", "medium")
                urls.append(url)
        return urls

    def download_image(self, url, dst_path):
        """Download single image."""
        try:
            r = self.session.get(url, timeout=20, stream=True)
            r.raise_for_status()
            with open(dst_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
        except Exception:
            return False

    def collect_species(self, species_name, taxon_id, text_query=None):
        """Collect images for one species.

        text_query: optional free-text filter (iNat observations `q` search on
        tags/description), combined with taxon_id. Used to target color-morphs
        or breeds that iNat has no dedicated taxon for (e.g. taxon=red_fox +
        text_query="melanistic" to find the dark color-morph specifically).
        """
        out_dir = self.output_dir / species_name
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))
        if existing >= self.n_per_species:
            print(f"  {species_name}: already have {existing} images, skipping")
            return existing

        qsuffix = f" q={text_query!r}" if text_query else ""
        print(f"  {species_name} (taxon={taxon_id}{qsuffix}): collecting {self.n_per_species} images...")

        is_urban_bear = (species_name == "bear_urban")
        if is_urban_bear:
            print(f"    (will filter for urban/suburban context from observation metadata)")

        URBAN_KEYWORDS = [
            "urban", "city", "town", "suburb", "neighborhood", "backyard",
            "road", "street", "parking", "dumpster", "garbage", "residential",
            "downtown", "industrial", "yard", "house", "school", "store",
            "highway", "campground", "camp", "cabin", "porch", "deck",
        ]

        collected = existing
        page = 1
        url_set = set()

        extra_params = None
        if text_query:
            extra_params = {"q": text_query, "search_on": "tags,description",
                            "quality_grade": "any"}

        while collected < self.n_per_species:
            data = self.search_observations(taxon_id, page=page, extra_params=extra_params)
            if not data or not data.get("results"):
                break

            obs_list = data["results"]
            if not obs_list:
                break

            for obs in obs_list:
                if collected >= self.n_per_species:
                    break

                # For bear_urban: filter locally using place_guess + description + tags
                if is_urban_bear:
                    desc  = (obs.get("description") or "").lower()
                    tags  = " ".join(obs.get("tag_list") or []).lower()
                    place = (obs.get("place_guess") or "").lower()
                    combined = f"{desc} {tags} {place}"
                    has_urban = any(kw in combined for kw in URBAN_KEYWORDS)
                    if not has_urban:
                        continue

                urls = self.get_photo_urls(obs)
                for url in urls[:1]:
                    if url in url_set:
                        continue
                    url_set.add(url)

                    obs_id = obs.get("id", "unknown")
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                    fname = f"{species_name}_{obs_id}_{url_hash}.jpg"
                    dst = out_dir / fname

                    if obs_id in self.exclude_obs:
                        continue
                    if dst.exists():
                        # already counted in `existing` above -- counting it
                        # again made a top-up stop at half its target
                        continue

                    if self.download_image(url, dst):
                        collected += 1
                        if collected % 50 == 0:
                            print(f"    {species_name}: {collected}/{self.n_per_species}")

            page += 1
            time.sleep(0.5)

            # Higher page limit for urban bear since many observations get filtered
            if page > (200 if is_urban_bear else 50):
                print(f"    {species_name}: reached page limit at {collected} images")
                break

        print(f"  {species_name}: collected {collected} images -> {out_dir}")
        return collected

    def run(self, species_list=None, text_species_list=None):
        """Collect all species. text_species_list uses INAT_TEXT_QUERIES: for each
        species, runs multiple free-text queries against the same taxon, accumulating
        into one output folder (each call continues from the existing on-disk count,
        so later queries top up whatever earlier queries didn't reach)."""
        if species_list is None and text_species_list is None:
            species_list = list(INAT_TAXA.keys())

        total = 0
        for species in (species_list or []):
            if species not in INAT_TAXA:
                print(f"  Unknown species: {species}")
                continue
            taxon_id = INAT_TAXA[species]
            n = self.collect_species(species, taxon_id)
            total += n
            time.sleep(1)

        for species in (text_species_list or []):
            if species not in INAT_TEXT_QUERIES:
                print(f"  Unknown text-species: {species}")
                continue
            taxon_id, queries = INAT_TEXT_QUERIES[species]
            n = 0
            for q in queries:
                n = self.collect_species(species, taxon_id, text_query=q)
                time.sleep(1)
            total += n

        print(f"\niNaturalist: collected {total} images total")
        return total


# ============================================================
# CLIP FILTER — keep only trail-cam-like images
# ============================================================
def clip_filter(image_dir, threshold=0.25, batch_size=64):
    """
    Use CLIP to filter images that look like trail camera footage.
    Removes studio/zoo/domestic photos.
    """
    try:
        import torch
        import clip
        from PIL import Image
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "git+https://github.com/openai/CLIP.git",
                        "--break-system-packages", "-q"])
        import torch
        import clip
        from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    prompts_keep = [
        "trail camera wildlife photo",
        "wildlife in forest",
        "wild animal in nature",
        "animal in woods at night",
        "security camera animal photo",
    ]
    prompts_reject = [
        "zoo animal",
        "pet animal indoors",
        "studio photo",
        "cartoon drawing",
        "person holding animal",
        "domestic animal",
    ]

    text_keep   = clip.tokenize(prompts_keep).to(device)
    text_reject = clip.tokenize(prompts_reject).to(device)

    with torch.no_grad():
        feat_keep   = model.encode_text(text_keep).float()
        feat_reject = model.encode_text(text_reject).float()
        feat_keep   /= feat_keep.norm(dim=-1, keepdim=True)
        feat_reject /= feat_reject.norm(dim=-1, keepdim=True)

    image_dir = Path(image_dir)
    images = list(image_dir.rglob("*.jpg")) + list(image_dir.rglob("*.png"))

    kept = rejected = 0
    for img_path in images:
        try:
            img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                feat_img = model.encode_image(img).float()
                feat_img /= feat_img.norm(dim=-1, keepdim=True)

            sim_keep   = (feat_img @ feat_keep.T).mean().item()
            sim_reject = (feat_img @ feat_reject.T).mean().item()

            if sim_keep < threshold or sim_reject > sim_keep:
                img_path.unlink()
                rejected += 1
            else:
                kept += 1
        except Exception:
            img_path.unlink()
            rejected += 1

    print(f"CLIP filter: kept={kept}, rejected={rejected} ({rejected/(kept+rejected)*100:.1f}% removed)")
    return kept


# ============================================================
# GEMINI VERIFY — confirm no bear in images
# ============================================================
def gemini_verify(image_dir, cache_file="inat_verify_cache.pkl", rpm=9):
    """
    Use Claude (Bedrock) to confirm images do NOT contain bears.
    Removes any image where Claude says YES (has bear).
    Cache stores species/dataset/answer for reuse in resplit and HEM verify.

    Cache format (same as lila_verify_cache.pkl):
    {
        str(img_path): {
            "has_bear": bool,
            "claude_answer": "YES"/"NO"/"ERROR",
            "species": "raccoon",
            "dataset": "inat",
            "verified": True,
            "gemini": "YES"/"NO"/"UNSURE",  # HEM-compatible key
            "folder_label": 0,
        }
    }
    """
    import io, pickle
    try:
        import boto3
        from PIL import Image
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "boto3", "Pillow", "--break-system-packages", "-q"])
        import boto3
        from PIL import Image

    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    )
    MODEL_ID = os.environ.get(
        "BEDROCK_MODEL_ID",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    PROMPT = "Is there a real, live bear in this image? Reply ONE WORD: YES or NO."

    # Load existing cache
    cache_path = Path(cache_file)
    cache = {}
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"  Loaded verify cache: {len(cache)} entries")

    def infer_metadata(img_path):
        p = Path(img_path)
        species = p.parent.name  # folder name = species
        return species, "inat"

    # This account's Bedrock quota is 10 requests/minute. Firing 10 threads at
    # once does not go faster -- it just converts requests into throttles and
    # then TIMEOUTs (a first run at 4 unpaced workers against this same quota
    # lost 17% of its answers that way, see export_confident_label_audit.py).
    # Pace submissions globally instead of relying on retry-after-throttle.
    lock = threading.Lock()
    next_slot = [0.0]
    interval = 60.0 / max(1, rpm)

    def pace():
        with lock:
            now = time.time()
            wait = max(0.0, next_slot[0] - now)
            next_slot[0] = max(now, next_slot[0]) + interval
        if wait > 0:
            time.sleep(wait)

    def check_one(img_path):
        fp_str = str(img_path)
        if fp_str in cache:
            return img_path, cache[fp_str]

        species, dataset = infer_metadata(img_path)
        retries, backoff = 4, 2
        while retries > 0:
            pace()
            try:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((512, 512))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                resp = client.converse(
                    modelId=MODEL_ID,
                    messages=[{"role": "user", "content": [
                        {"image": {"format": "jpeg", "source": {"bytes": buf.getvalue()}}},
                        {"text": PROMPT}
                    ]}],
                    inferenceConfig={"maxTokens": 10, "temperature": 0.0}
                )
                ans = resp['output']['message']['content'][0]['text'].strip().upper()
                has_bear = "YES" in ans
                result = {
                    "has_bear": has_bear,
                    "claude_answer": "YES" if has_bear else "NO",
                    "species": species,
                    "dataset": dataset,
                    "verified": True,
                    "gemini": "YES" if has_bear else "NO",
                    "folder_label": 0,
                }
                return img_path, result
            except Exception as e:
                err = str(e).lower()
                if "throttling" in err or "429" in err or "503" in err:
                    time.sleep(backoff + random.uniform(1, 2))
                    retries -= 1; backoff *= 2
                else:
                    break
        result = {
            "has_bear": False,
            "claude_answer": "ERROR",
            "species": species,
            "dataset": dataset,
            "verified": False,
            "gemini": "UNSURE",
            "folder_label": 0,
        }
        return img_path, result

    image_dir = Path(image_dir)
    all_images = list(image_dir.rglob("*.jpg")) + list(image_dir.rglob("*.png"))
    uncached = [p for p in all_images if str(p) not in cache]

    print(f"\nClaude Bedrock verify: {len(all_images)} images "
          f"({len(all_images)-len(uncached)} cached, {len(uncached)} new)")

    from collections import Counter
    species_count = Counter(Path(p).parent.name for p in all_images)
    print("  Species breakdown:")
    for sp, cnt in sorted(species_count.items()):
        print(f"    {sp:<20}: {cnt}")

    removed = kept = processed = 0

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(check_one, p): p for p in all_images}
        for fut in as_completed(futures):
            img_path, result = fut.result()
            fp_str = str(img_path)
            cache[fp_str] = result

            if result["has_bear"]:
                if img_path.exists():
                    img_path.unlink()
                removed += 1
            else:
                kept += 1

            processed += 1
            if processed % 100 == 0:
                with open(cache_path, "wb") as f:
                    pickle.dump(cache, f)
                print(f"  {processed}/{len(all_images)} — kept={kept}, removed={removed}")

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"\n  Results by species:")
    species_kept = Counter()
    species_removed = Counter()
    for fp_str, info in cache.items():
        sp = info.get("species", "unknown")
        if info["has_bear"]:
            species_removed[sp] += 1
        else:
            species_kept[sp] += 1
    for sp in sorted(set(list(species_kept.keys()) + list(species_removed.keys()))):
        k = species_kept.get(sp, 0)
        r = species_removed.get(sp, 0)
        print(f"    {sp:<20}: kept={k}, removed={r}")

    print(f"\n  Total: kept={kept}, removed (had bear)={removed}")
    print(f"  Cache saved: {len(cache)} entries -> {cache_path}")
    print(f"  Cache reusable for: resplit stratification, HEM verify")
    return kept


# ============================================================
# MERGE TO DATASET
# ============================================================
def merge_to_dataset(source_dir, dataset_non_bear_dir, copy=True):
    """Copy/move collected images into dataset non_bear folder."""
    source_dir = Path(source_dir)
    dst_dir = Path(dataset_non_bear_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    images = list(source_dir.rglob("*.jpg")) + list(source_dir.rglob("*.png"))
    moved = skipped = 0

    for src in images:
        # Preserve species subfolder in filename
        species = src.parent.name
        dst = dst_dir / f"{species}_{src.name}"
        if dst.exists():
            skipped += 1
            continue
        if copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))
        moved += 1

    print(f"Merged: {moved} images -> {dst_dir} ({skipped} skipped, duplicates)")
    return moved


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Collect bear-confusable animals")
    parser.add_argument("--source", choices=["inat", "lila", "both"], default="inat")
    parser.add_argument("--output", default="non_bear_confusable",
                        help="Output directory for collected images")
    parser.add_argument("--n", type=int, default=300,
                        help="Images per species (default: 300)")
    parser.add_argument("--species", nargs="*",
                        default=["elk", "moose", "raccoon", "wild_boar", "coyote"],
                        help="Species to collect (taxon-only, see INAT_TAXA). "
                             "Pass --species with no values to skip taxon-only collection "
                             "(e.g. when only using --text-species).")
    parser.add_argument("--text-species", nargs="*", default=[],
                        help="Text-filtered species to collect, see INAT_TEXT_QUERIES "
                             "(e.g. dark_dog, dark_fox) — runs multiple free-text queries "
                             "per species, constrained by taxon, to target breeds/color-morphs "
                             "iNat has no dedicated taxon for.")
    parser.add_argument("--clip-filter", action="store_true",
                        help="Filter non-trail-cam images with CLIP")
    parser.add_argument("--gemini-verify", action="store_true",
                        help="Remove any images containing bears via Claude Bedrock")
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip download, only run verify on existing images in --output")
    parser.add_argument("--merge-to",
                        help="After collecting, merge into this non_bear dataset folder")
    parser.add_argument("--exclude-obs-from", nargs="*", default=[],
                        help="dirs whose <species>_<obsid>[_hash].jpg files name "
                             "observations to skip (train + held-out test)")
    parser.add_argument("--quality", default="research",
                        choices=["research", "needs_id", "casual"],
                        help="iNaturalist observation quality grade")
    parser.add_argument("--rpm", type=int, default=9,
                        help="Bedrock requests/minute for --gemini-verify "
                             "(this account's quota is 10 RPM on Sonnet 4.5; "
                             "default leaves 1 RPM of margin)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Collecting bear-confusable animals")
    print(f"  Species: {args.species}")
    print(f"  Text-species: {args.text_species}")
    print(f"  N per species: {args.n}")
    print(f"  Output: {output_dir}")
    print(f"  Mode: {'verify-only' if args.verify_only else 'download' + (' + verify' if args.gemini_verify else '')}")
    print(f"{'='*60}\n")

    total = 0

    if not args.verify_only:
        if args.source in ("inat", "both"):
            print("--- iNaturalist ---")
            exclude = set()
            for d in args.exclude_obs_from:
                for f in Path(d).rglob("*.jpg"):
                    parts = f.stem.split("_")
                    for tok in parts[1:]:          # species names contain "_"
                        if tok.isdigit() and len(tok) >= 4:
                            exclude.add(int(tok))
                            break
            if exclude:
                print(f"  excluding {len(exclude):,} observation ids already in train/test")
            collector = INatCollector(output_dir, n_per_species=args.n, quality=args.quality,
                                      exclude_obs=exclude)
            n = collector.run(args.species, args.text_species)
            total += n

        if args.source in ("lila", "both"):
            print("\n--- LILA BC ---")
            print("LILA BC requires manual dataset download.")
            print("Visit: https://lila.science/datasets/")
            for name, info in LILA_DATASETS.items():
                print(f"  {name}: {info['url']}")
                print(f"    Species: {', '.join(info['species'])}")

        print(f"\nTotal collected: {total} images")

        if args.clip_filter and total > 0:
            print("\n--- CLIP Filter ---")
            kept = clip_filter(output_dir)
            print(f"After CLIP filter: {kept} images remaining")

    else:
        # verify-only: count existing images
        existing = list(output_dir.rglob("*.jpg")) + list(output_dir.rglob("*.png"))
        total = len(existing)
        print(f"Verify-only mode: found {total} existing images in {output_dir}")

    if args.gemini_verify or args.verify_only:
        print("\n--- Claude Bedrock Bear Verification ---")
        kept = gemini_verify(output_dir, rpm=args.rpm)
        print(f"After verification: {kept} images remaining")

    if args.merge_to:
        print(f"\n--- Merging to dataset ---")
        n = merge_to_dataset(output_dir, args.merge_to)
        print(f"Done. Run merge_new_animals.py to rebuild splits.")

    print(f"\n{'='*60}")
    print(f"Next steps:")
    print(f"  1. Review images in {output_dir}/")
    print(f"  2. python merge_new_animals.py --dataset bear_dataset_v_final_final \\")
    print(f"         --nonbear-dir {output_dir} --cache inat_verify_cache.pkl")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
