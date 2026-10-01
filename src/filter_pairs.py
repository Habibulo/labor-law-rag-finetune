"""Phase 3.3a: rule, duplicate, and round-trip scoring of synthetic (question, chunk) pairs.

Usage:
    python src/filter_pairs.py          # then: python src/judge_pairs.py

Steps, each counted in results/phase3_filter/stats.json:
  1. Rules: too short, not Korean, Chinese characters, copied article numbers.
  2. Exact and near-duplicate questions (char 3-gram Jaccard). A duplicate across splits keeps the
     test/val copy and drops the train copy, so no test question is effectively trained on.
  3. Round-trip ranks of the source article/case under BM25 (Kiwi) and the base embedding model.
     Recorded only (decisions_log #33); answerability is judged in judge_pairs.py.

Output: data/synthetic/queries_scored.jsonl (every question + ranks + drop_reason),
        results/phase3_filter/{stats.json, roundtrip_fail_sample.md}
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import yaml

from common import ROOT, read_jsonl
from retrievers import BM25Kiwi, Dense, group_ranks

sys.stdout.reconfigure(encoding="utf-8")

RAW = ROOT / "data" / "synthetic" / "queries_raw.jsonl"
SCORED = ROOT / "data" / "synthetic" / "queries_scored.jsonl"
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
RESULTS = ROOT / "results" / "phase3_filter"
CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
ARTICLE_RE = re.compile(r"제\s*\d+\s*조")


def norm(s):
    return re.sub(r"[\s\W_]+", "", s)


def ngrams(s, n):
    s = norm(s)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def hangul_ratio(s):
    letters = [c for c in s if c.isalpha()]
    return sum(0xAC00 <= ord(c) <= 0xD7A3 for c in letters) / max(1, len(letters))


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "filter.yaml").read_text(encoding="utf-8"))
    qs = read_jsonl(RAW)
    chunks = read_jsonl(CHUNKS)
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    for q in qs:
        q["drop_reason"] = None

    # 1. Rules
    r = cfg["rules"]
    for q in qs:
        if len(norm(q["question"])) < r["min_chars"]:
            q["drop_reason"] = "too_short"
        elif hangul_ratio(q["question"]) < r["min_hangul_ratio"]:
            q["drop_reason"] = "not_korean"
        elif r["drop_if_chinese_chars"] and CJK_RE.search(q["question"]):
            q["drop_reason"] = "chinese_chars"
        elif r["drop_if_article_number"] and ARTICLE_RE.search(q["question"]):
            q["drop_reason"] = "article_number"

    # 2. Duplicates: exact (normalized) and near (char n-gram Jaccard), priority test > val > train
    nd = cfg["near_duplicate"]
    prio = {s: i for i, s in enumerate(nd["split_priority"])}
    alive = sorted([q for q in qs if not q["drop_reason"]], key=lambda q: (prio[q["split"]], q["qid"]))
    grams = [ngrams(q["question"], nd["char_ngram"]) for q in alive]
    lengths = [len(g) for g in grams]
    kept_idx = []
    for i, q in enumerate(alive):
        dup_of = None
        for j in kept_idx:
            # Jaccard >= t requires the smaller set to be >= t * larger set; skip impossible pairs fast.
            lo, hi = sorted((lengths[i], lengths[j]))
            if lo < nd["jaccard_threshold"] * hi:
                continue
            inter = len(grams[i] & grams[j])
            if inter / (lengths[i] + lengths[j] - inter) >= nd["jaccard_threshold"]:
                dup_of = alive[j]
                break
        if dup_of is None:
            kept_idx.append(i)
        else:
            q["drop_reason"] = "duplicate" if dup_of["split"] == q["split"] else f"duplicate_of_{dup_of['split']}"
            q["duplicate_of"] = dup_of["qid"]

    # 3. Round-trip ranks (information only)
    rt = cfg["round_trip"]
    alive = [q for q in qs if not q["drop_reason"]]
    texts = [c["text"] for c in chunks]
    chunk_groups = [c["group_id"] for c in chunks]
    targets = [q["group_id"] for q in alive]
    questions = [q["question"] for q in alive]
    print("BM25 (Kiwi) ranking ...")
    bm25_ranks = group_ranks(BM25Kiwi(texts).scores(questions), chunk_groups, targets)
    print(f"Dense ranking with {rt['dense_model']} ...")
    dense = Dense(rt["dense_model"], rt["query_prefix"], rt["passage_prefix"])
    dense_ranks = group_ranks(dense.scores(questions, dense.encode_corpus(texts)), chunk_groups, targets)
    for q, br, dr in zip(alive, bm25_ranks, dense_ranks):
        q["bm25_rank"], q["dense_rank"] = int(br), int(dr)
        q["round_trip_pass"] = bool(min(br, dr) <= rt["top_k"])

    counts = Counter(total=len(qs))
    by_split = defaultdict(Counter)
    for q in qs:
        counts[f"drop_{q['drop_reason']}" if q["drop_reason"] else "candidates"] += 1
        by_split[q["split"]]["candidate" if not q["drop_reason"] else q["drop_reason"]] += 1
    k = rt["top_k"]
    summary = {
        "counts": dict(counts),
        "by_split": {s: dict(c) for s, c in sorted(by_split.items())},
        f"round_trip_pass_rate_top{k}": round(float(np.mean([q["round_trip_pass"] for q in alive])), 4),
        "base_recall_at_1": {"bm25": round(float(np.mean(bm25_ranks == 1)), 4),
                             "dense": round(float(np.mean(dense_ranks == 1)), 4)},
        f"base_recall_at_{k}": {"bm25": round(float(np.mean(bm25_ranks <= k)), 4),
                                "dense": round(float(np.mean(dense_ranks <= k)), 4)},
        "note": "Recall here is over all candidate synthetic questions (all splits); indicative only, "
                "not the Phase 4 baseline (the test set is not human-verified yet).",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with SCORED.open("w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    fails = [q for q in alive if not q["round_trip_pass"] and q["split"] != "test"]
    sample = random.Random(cfg["seed"]).sample(fails, min(cfg["inspect_dropped"], len(fails)))
    lines = ["# 20 random train/val pairs that FAIL the round-trip check (top 20)", "",
             "Manual check 2026-09-27: 19 of these 20 were correct pairs -> round-trip is recorded only.", ""]
    for q in sample:
        c = chunk_by_id[q["chunk_id"]]
        lines += [f"## {q['qid']}", f"- Question: {q['question']}", f"- Source: {c['title']}",
                  f"- BM25 rank: {q['bm25_rank']}, dense rank: {q['dense_rank']}",
                  f"- Source text: {c['text'][:300].replace(chr(10), ' ')} ...", ""]
    (RESULTS / "roundtrip_fail_sample.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote {SCORED}. Next: python src/judge_pairs.py")


if __name__ == "__main__":
    main()
