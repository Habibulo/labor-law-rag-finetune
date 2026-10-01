"""Phase 4/5: evaluate retrievers on the held-out test queries.

Usage:
    python src/eval_retrieval.py                       # all baselines
    python src/eval_retrieval.py --model path/to/ft --name e5-finetuned
    python src/eval_retrieval.py --split val

A query counts as answered when any chunk of its SOURCE article/case (group_id) is retrieved,
because an article split across chunks is one legal unit. Metrics come with 95% bootstrap
confidence intervals so small differences are not over-read.

Output: results/phase4_baselines/{metrics.json, comparison.png}
"""
import argparse
import json
import sys

import matplotlib
import numpy as np
import yaml

from common import ROOT, read_jsonl
from retrievers import BM25Kiwi, Dense, group_ranks

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
QUERIES = ROOT / "data" / "synthetic" / "queries_filtered.jsonl"
RESULTS = ROOT / "results" / "phase4_baselines"
KS = (1, 5, 10)


def metrics_from_ranks(ranks):
    """ranks: 1-based rank of the first correct chunk, per query."""
    ranks = np.asarray(ranks, dtype=float)
    out = {f"recall@{k}": float(np.mean(ranks <= k)) for k in KS}
    out["mrr@10"] = float(np.mean(np.where(ranks <= 10, 1.0 / ranks, 0.0)))
    # Single relevant group per query, so DCG = 1/log2(rank+1) and the ideal DCG is 1.
    out["ndcg@10"] = float(np.mean(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0)))
    return out


def bootstrap_ci(ranks, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    ranks = np.asarray(ranks)
    draws = [metrics_from_ranks(ranks[rng.integers(0, len(ranks), len(ranks))]) for _ in range(n)]
    return {k: [float(np.percentile([d[k] for d in draws], 2.5)),
                float(np.percentile([d[k] for d in draws], 97.5))] for k in draws[0]}


def rrf(score_lists, k=60):
    """Reciprocal rank fusion of several score matrices."""
    total = np.zeros_like(score_lists[0])
    for s in score_lists:
        order = np.argsort(-s, axis=1)
        rank = np.empty_like(order)
        rows = np.arange(s.shape[0])[:, None]
        rank[rows, order] = np.arange(s.shape[1])[None, :]
        total += 1.0 / (k + rank + 1)
    return total


def evaluate(name, ranks, store):
    m = metrics_from_ranks(ranks)
    store[name] = {**m, "ci95": bootstrap_ci(ranks), "n_queries": len(ranks)}
    print(f"{name:28s} R@1 {m['recall@1']:.3f}  R@5 {m['recall@5']:.3f}  "
          f"R@10 {m['recall@10']:.3f}  MRR@10 {m['mrr@10']:.3f}  NDCG@10 {m['ndcg@10']:.3f}")
    return store[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=None, help="extra dense model path/name to evaluate")
    ap.add_argument("--name", default=None, help="label for --model")
    ap.add_argument("--skip-bge", action="store_true", help="skip bge-m3 (2.2GB download)")
    ap.add_argument("--keep-leaked", action="store_true",
                    help="keep test queries whose chunk text also appears in train (default: drop)")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "filter.yaml").read_text(encoding="utf-8"))
    chunks = read_jsonl(CHUNKS)
    texts = [c["text"] for c in chunks]
    groups = [c["group_id"] for c in chunks]
    qs = [q for q in read_jsonl(QUERIES) if q["split"] == args.split]
    # Upstream stores a few judgments twice under different 판례일련번호, so identical text can sit
    # in two splits. Those queries are dropped: their answer text was seen in training.
    by_id = {c["chunk_id"]: c for c in chunks}
    train_texts = {by_id[q["chunk_id"]]["text"] for q in read_jsonl(QUERIES) if q["split"] == "train"}
    leaked = [q for q in qs if by_id[q["chunk_id"]]["text"] in train_texts]
    if leaked and not args.keep_leaked:
        qs = [q for q in qs if q not in leaked]
        print(f"excluded {len(leaked)} {args.split} queries whose chunk text also appears in train "
              f"(duplicate upstream records): {[q['qid'] for q in leaked]}")
    questions = [q["question"] for q in qs]
    targets = [q["group_id"] for q in qs]
    print(f"{len(qs)} {args.split} queries over {len(chunks)} chunks "
          f"({len({g for g in groups})} articles/cases)\n")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS / "metrics.json"
    store = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else {}
    store.setdefault("_meta", {})["split"] = args.split
    store["_meta"]["n_chunks"] = len(chunks)
    store["_meta"]["n_queries"] = len(qs)
    store["_meta"]["excluded_duplicate_text_queries"] = [q["qid"] for q in leaked]
    store["_meta"]["note"] = ("Test queries are synthetic (LLM-generated) and NOT human-verified; "
                             "Phase 3.5 human review was not run. Before/after comparisons are still "
                             "valid because every system sees the identical query set.")

    scores = {}
    print("BM25 (Kiwi morphemes) ...")
    scores["bm25"] = BM25Kiwi(texts).scores(questions)
    evaluate("BM25 (Kiwi)", group_ranks(scores["bm25"], groups, targets), store)

    dense_specs = [("multilingual-e5-base", "intfloat/multilingual-e5-base", "query: ", "passage: ")]
    if not args.skip_bge:
        dense_specs.append(("bge-m3", "BAAI/bge-m3", "", ""))
    if args.model:
        dense_specs.append((args.name or args.model, args.model, "query: ", "passage: "))

    for label, path, qp, pp in dense_specs:
        print(f"\n{label} ...")
        d = Dense(path, qp, pp)
        scores[label] = d.scores(questions, d.encode_corpus(texts))
        evaluate(label, group_ranks(scores[label], groups, targets), store)
        del d

    best_dense = max([k for k in scores if k != "bm25"],
                     key=lambda k: store[k]["recall@10"], default=None)
    if best_dense:
        print(f"\nHybrid RRF (BM25 + {best_dense}) ...")
        fused = rrf([scores["bm25"], scores[best_dense]])
        evaluate(f"Hybrid RRF (BM25+{best_dense})", group_ranks(fused, groups, targets), store)

    out_file.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")

    names = [k for k in store if not k.startswith("_")]
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(names)), 5))
    x = np.arange(len(names))
    for i, metric in enumerate(["recall@1", "recall@5", "recall@10", "mrr@10"]):
        vals = [store[n][metric] for n in names]
        lo = [store[n][metric] - store[n]["ci95"][metric][0] for n in names]
        hi = [store[n]["ci95"][metric][1] - store[n][metric] for n in names]
        ax.bar(x + i * 0.2 - 0.3, vals, 0.2, yerr=[lo, hi], capsize=2, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("score")
    ax.set_title(f"Retrieval on {len(qs)} held-out {args.split} queries (95% bootstrap CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison.png", dpi=120)
    print(f"\nWrote {out_file} and {RESULTS / 'comparison.png'}")


if __name__ == "__main__":
    main()
