"""Phase 5 (Track A): fine-tune the embedding model on labor-law question/article pairs.

Usage:
    python src/train_embedding.py

Trains ONLY on train-split pairs. Val pairs steer checkpoint selection; the test split is never
seen. Loss is MultipleNegativesRankingLoss (cached), which pulls a question towards its own
article and pushes it away from every other article in the batch.

Output: models/e5-base-labor-law/, results/phase5_embedding/train_meta.json
"""
import json
import random
import sys
import time

import numpy as np
import torch
import yaml
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from common import ROOT, read_jsonl

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
QUERIES = ROOT / "data" / "synthetic" / "queries_filtered.jsonl"
RESULTS = ROOT / "results" / "phase5_embedding"


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "train_embedding.yaml").read_text(encoding="utf-8"))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    chunks = {c["chunk_id"]: c for c in read_jsonl(CHUNKS)}
    qs = read_jsonl(QUERIES)
    qp, pp = cfg["query_prefix"], cfg["passage_prefix"]

    train = [q for q in qs if q["split"] == "train"]
    val = [q for q in qs if q["split"] == "val"]
    ds = Dataset.from_dict({
        "anchor": [qp + q["question"] for q in train],
        "positive": [pp + chunks[q["chunk_id"]]["text"] for q in train],
    }).shuffle(seed=cfg["seed"])
    print(f"train pairs: {len(train)} | val pairs: {len(val)}")

    # Val IR evaluator: retrieve over the val chunks, correct if the source chunk is found.
    val_corpus = {cid: pp + c["text"] for cid, c in chunks.items()
                  if any(v["chunk_id"] == cid for v in val)}
    val_queries = {q["qid"]: qp + q["question"] for q in val}
    relevant = {q["qid"]: {q["chunk_id"]} for q in val}
    evaluator = InformationRetrievalEvaluator(val_queries, val_corpus, relevant, name="val",
                                              show_progress_bar=False)

    model = SentenceTransformer(cfg["base_model"])
    model.max_seq_length = cfg["max_seq_length"]
    base = evaluator(model)
    print("base model on val:", {k: round(v, 4) for k, v in base.items() if "cosine" in k})

    out = ROOT / cfg["output_dir"]
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "_checkpoints"),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        fp16=cfg["fp16"],
        batch_sampler=BatchSamplers.NO_DUPLICATES,   # no repeated passage inside one batch
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_cosine_ndcg@10",
        logging_steps=20,
        seed=cfg["seed"],
        report_to=[],
    )
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=cfg["mini_batch_size"])
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss,
                                         evaluator=evaluator)
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60
    model.save_pretrained(str(out))

    after = evaluator(model)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    meta = {
        "config": cfg,
        "n_train_pairs": len(train),
        "n_val_pairs": len(val),
        "train_minutes": round(minutes, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "peak_gpu_memory_gb": round(peak_gb, 2) if peak_gb else None,
        "val_before": {k: round(v, 4) for k, v in base.items()},
        "val_after": {k: round(v, 4) for k, v in after.items()},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"\ntrained {minutes:.1f} min on {meta['gpu']} (peak {meta['peak_gpu_memory_gb']} GB)")
    print("val ndcg@10 before -> after:",
          base.get("val_cosine_ndcg@10"), "->", after.get("val_cosine_ndcg@10"))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
