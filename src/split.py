"""Phase 3.1: assign every chunk group (one article or one precedent) to train/val/test.

Usage:
    python src/split.py

All chunks of one article (its 항-splits) or one precedent share a group_id and always land in
the same split, so no test question's source text is ever used to generate training questions.
Split is stratified by doc_type and seeded.

Output: data/processed/splits.json {group_id: split}, results/phase3_split/stats.json
"""
import json
import random
import sys
from collections import Counter, defaultdict

import yaml

from common import ROOT, read_jsonl

sys.stdout.reconfigure(encoding="utf-8")

CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
OUT = ROOT / "data" / "processed" / "splits.json"
RESULTS = ROOT / "results" / "phase3_split"


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "queries.yaml").read_text(encoding="utf-8"))
    chunks = read_jsonl(CHUNKS)
    groups = defaultdict(list)
    for c in chunks:
        groups[c["group_id"]].append(c)

    force_key = cfg["split"]["force_train_if"]
    by_type = defaultdict(list)
    forced = []
    for gid, cs in sorted(groups.items()):
        if any(c["metadata"].get(force_key) for c in cs):
            forced.append(gid)
        else:
            by_type[cs[0]["doc_type"]].append(gid)

    rng = random.Random(cfg["seed"])
    assign = {gid: "train" for gid in forced}
    for doc_type, gids in sorted(by_type.items()):
        rng.shuffle(gids)
        n_test = round(len(gids) * cfg["split"]["test_ratio"])
        n_val = round(len(gids) * cfg["split"]["val_ratio"])
        for i, gid in enumerate(gids):
            assign[gid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    OUT.write_text(json.dumps(assign, ensure_ascii=False, indent=0), encoding="utf-8")

    stats = {"groups": defaultdict(Counter), "chunks": defaultdict(Counter), "forced_train_groups": len(forced)}
    for gid, cs in groups.items():
        stats["groups"][cs[0]["doc_type"]][assign[gid]] += 1
        stats["chunks"][cs[0]["doc_type"]][assign[gid]] += len(cs)
    for k in ("groups", "chunks"):
        stats[k] = {t: dict(c) for t, c in sorted(stats[k].items())}
        stats[k]["total"] = dict(sum((Counter(v) for v in stats[k].values()), Counter()))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
