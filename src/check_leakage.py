"""Leakage audit: prove the test numbers are not inflated by training on test material.

Usage:
    python src/check_leakage.py

Checks, all of which must pass before any number is reported:
  1. No article/case (group_id) appears in both the train and the test split.
  2. No test chunk text was used as a training positive.
  3. No test question is a near-duplicate of a training question (char 3-gram Jaccard).
  4. The test questions evaluated are exactly the held-out split, none seen in training.

Output: results/phase4_baselines/leakage_check.json
"""
import json
import re
import sys
from collections import Counter

from common import ROOT, read_jsonl

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

RESULTS = ROOT / "results" / "phase4_baselines"


def ngrams(s, n=3):
    s = re.sub(r"[\s\W_]+", "", s)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def main():
    chunks = {c["chunk_id"]: c for c in read_jsonl(ROOT / "data/processed/chunks.jsonl")}
    qs = read_jsonl(ROOT / "data/synthetic/queries_filtered.jsonl")
    splits = json.loads((ROOT / "data/processed/splits.json").read_text(encoding="utf-8"))
    train = [q for q in qs if q["split"] == "train"]
    test = [q for q in qs if q["split"] == "test"]

    train_groups = {q["group_id"] for q in train}
    test_groups = {q["group_id"] for q in test}
    overlap_groups = train_groups & test_groups

    train_chunks = {q["chunk_id"] for q in train}
    test_chunks = {q["chunk_id"] for q in test}
    overlap_chunks = train_chunks & test_chunks

    # Exact chunk TEXT overlap, in case two different ids carry identical text.
    train_texts = {chunks[c]["text"] for c in train_chunks}
    overlap_text = sum(1 for c in test_chunks if chunks[c]["text"] in train_texts)

    train_grams = [ngrams(q["question"]) for q in train]
    train_lens = [len(g) for g in train_grams]
    near_dupes = []
    for q in test:
        g = ngrams(q["question"])
        for tg, tl in zip(train_grams, train_lens):
            lo, hi = sorted((len(g), tl))
            if lo < 0.8 * hi:
                continue
            inter = len(g & tg)
            if inter / (len(g) + tl - inter) >= 0.8:
                near_dupes.append(q["qid"])
                break

    mislabeled = [q["qid"] for q in test if splits[q["group_id"]] != "test"]

    report = {
        "n_train_queries": len(train), "n_test_queries": len(test),
        "train_groups": len(train_groups), "test_groups": len(test_groups),
        "group_overlap": len(overlap_groups),
        "chunk_id_overlap": len(overlap_chunks),
        "identical_chunk_text_overlap": overlap_text,
        "test_questions_near_duplicate_of_train": len(near_dupes),
        "test_queries_not_in_test_split": len(mislabeled),
        "test_doc_types": dict(Counter(q["doc_type"] for q in test)),
    }
    checks = {
        "no shared article/case between train and test": len(overlap_groups) == 0,
        "no shared chunk id": len(overlap_chunks) == 0,
        "no identical chunk text": overlap_text == 0,
        "no near-duplicate questions": len(near_dupes) == 0,
        "all test queries are in the test split": len(mislabeled) == 0,
    }
    report["checks"] = checks
    report["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "leakage_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{report['verdict']}")
    if report["verdict"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
