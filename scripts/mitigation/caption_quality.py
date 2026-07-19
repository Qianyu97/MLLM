"""Caption-quality metrics (BLEU-4 / CIDEr / ROUGE-L + object coverage) for
vanilla vs detect-then-revise captions, vs COCO GT captions. Proves the revise
does NOT degrade caption quality."""
import os
import sys, json, re
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
from cmpsa import paths
from cmpsa.utils import load_json
from cmpsa.eval.eval_chair import _extract_objects, _load_gt_objects
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

CAPS = os.path.expandvars(r"${CMPSA_DATA_ROOT}\cmpsa_project\results\predictions\detect_revise_caps.jsonl")

def tok(s):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())

rows = [json.loads(l) for l in open(CAPS, encoding="utf-8")]
print(f"loaded {len(rows)} caption pairs")

# COCO GT references
caps = load_json(paths.COCO_CAPTIONS_VAL2014)
refs = {}
for a in caps["annotations"]:
    refs.setdefault(int(a["image_id"]), []).append(tok(a["caption"]))

gts, res_v, res_r = {}, {}, {}
for r in rows:
    iid = int(r["image_id"])
    if iid not in refs: continue
    gts[iid] = refs[iid]
    res_v[iid] = [tok(r["vanilla"])]
    res_r[iid] = [tok(r["revised"])]

def score(res):
    b, _ = Bleu(4).compute_score(gts, res)
    c, _ = Cider().compute_score(gts, res)
    ro, _ = Rouge().compute_score(gts, res)
    return b[3], c, ro   # BLEU-4, CIDEr, ROUGE-L

bv, cv, rv = score(res_v)
br, cr, rr = score(res_r)

# object coverage/recall vs GT objects
gt_objects, _ = _load_gt_objects()
def cov(field):
    tot = mat = 0
    for r in rows:
        gold = gt_objects.get(int(r["image_id"]), set())
        if not gold: continue
        ment = _extract_objects(r[field])
        tot += len(gold); mat += len(set(ment) & gold)
    return mat / tot if tot else 0

print(f"\n{'metric':12s} {'vanilla':>10s} {'revised':>10s} {'delta':>10s}")
for name, a, b in [("BLEU-4", bv, br), ("CIDEr", cv, cr), ("ROUGE-L", rv, rr),
                   ("ObjCoverage", cov("vanilla"), cov("revised"))]:
    print(f"{name:12s} {a:10.4f} {b:10.4f} {b-a:+10.4f}")
