"""Build HalluProbe-VL OBJECT probes from COCO2017-val (leak-free vs POPE's 2014).
Balanced yes/no existence probes with HIGH-CO-OCCURRENCE hard negatives (POPE-
adversarial style). GT from COCO instances (reliable), NOT tool pseudo-labels.
Reuses derived/vg_attr + derived/vg_rel as the attribute/relation layers.
Writes real probes + honest stats.json (replacing the fabricated planned counts)."""
import sys, os, json, random
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
from collections import defaultdict, Counter
from cmpsa import paths
from cmpsa.utils import load_json

random.seed(42)
OUT = paths.HALLUPROBE
(OUT/"probes").mkdir(parents=True, exist_ok=True)

def resolve_val2017(iid):
    name = f"{int(iid):012d}.jpg"
    for root in (paths.COCO_VAL2017, r"G:\cmpsa_data\basic\coco\images\val2017"):
        p = os.path.join(str(root), name)
        if os.path.exists(p): return p
    return None

inst = load_json(paths.COCO_INSTANCES_VAL2017)
id2name = {c["id"]: c["name"] for c in inst["categories"]}
CATS = sorted(id2name.values())
img_cats = defaultdict(set)
for a in inst["annotations"]:
    nm = id2name.get(a["category_id"])
    if nm: img_cats[a["image_id"]].add(nm)

# co-occurrence for hard negatives: how often cat co-occurs with each present cat
cooc = defaultdict(Counter)
for cats in img_cats.values():
    for a in cats:
        for b in cats:
            if a != b: cooc[a][b] += 1

# only images whose file is available
avail = [iid for iid in img_cats if resolve_val2017(iid)]
random.shuffle(avail)
avail = avail[:2000]
print(f"COCO2017-val images available: {len(avail)}")

probes = []
qid = 0
for iid in avail:
    present = list(img_cats[iid])
    absent = [c for c in CATS if c not in img_cats[iid]]
    # positives: up to 3 present
    pos = random.sample(present, min(3, len(present)))
    # hard negatives: absent cats most co-occurring with present ones
    scores = Counter()
    for p in present:
        for c, n in cooc[p].items():
            if c in img_cats[iid]: continue
            scores[c] += n
    hard = [c for c, _ in scores.most_common(20)]
    neg_pool = hard if hard else absent
    neg = random.sample(neg_pool, min(len(pos), len(neg_pool)))
    imgpath = resolve_val2017(iid)
    for c in pos:
        qid += 1; probes.append({"id": qid, "image": imgpath, "text": f"Is there a {c} in the image?", "label": "yes", "type": "object", "source": "coco2017val"})
    for c in neg:
        qid += 1; probes.append({"id": qid, "image": imgpath, "text": f"Is there a {c} in the image?", "label": "no", "type": "object", "source": "coco2017val_hardneg"})

random.shuffle(probes)
with open(OUT/"probes"/"object.jsonl", "w", encoding="utf-8") as f:
    for p in probes: f.write(json.dumps(p)+"\n")
npos = sum(1 for p in probes if p["label"]=="yes"); nneg = len(probes)-npos
print(f"object probes: {len(probes)} (pos {npos} / neg {nneg}) over {len(avail)} images")

# reuse VG-Attr / VG-Rel counts for the honest stats
def count_jsonl(p):
    try: return sum(1 for _ in open(p, encoding="utf-8"))
    except Exception: return 0
n_attr = count_jsonl(paths.VG_ATTR_JSONL); n_rel = count_jsonl(paths.VG_REL_JSONL)

stats = {
    "note": "REAL counts (replaces the prior fabricated planned label counts).",
    "object": {"n_probes": len(probes), "pos": npos, "neg": nneg,
               "source": "COCO2017-val instances GT; high-co-occurrence hard negatives; leak-free vs POPE(2014)",
               "images": len(avail)},
    "attribute": {"n_probes": n_attr, "source": "derived/vg_attr (Visual Genome attributes GT)"},
    "relation": {"n_probes": n_rel, "source": "derived/vg_rel (Visual Genome relationships GT)"},
    "gt_provenance": "COCO/VG source annotations (reliable), NOT tool pseudo-labels; tools (G-DINO/CLIP) are the DETECTORS evaluated.",
}
with open(OUT/"annotations"/"stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print("stats:", json.dumps(stats, ensure_ascii=False))
