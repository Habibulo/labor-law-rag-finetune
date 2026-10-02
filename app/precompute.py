"""Precompute corpus embeddings and the chunk table the Gradio demo serves.

Usage (run locally, with the GPU):
    python app/precompute.py

The free Spaces CPU would need minutes to embed 3,922 chunks on every restart, so the corpus
side is computed once here and shipped with the Space. Only the user's question is encoded live.

Output: app/assets/{chunks.json, emb_base.npy, emb_finetuned.npy}  (~12 MB each .npy)
"""
import json
import sys

import numpy as np
import yaml

from common import ROOT, read_jsonl
from retrievers import Dense

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

OUT = ROOT / "app" / "assets"


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "train_embedding.yaml").read_text(encoding="utf-8"))
    chunks = read_jsonl(ROOT / "data" / "processed" / "chunks.jsonl")
    texts = [c["text"] for c in chunks]
    OUT.mkdir(parents=True, exist_ok=True)

    table = [{"chunk_id": c["chunk_id"], "title": c["title"], "doc_type": c["doc_type"],
              "source_url": c["source_url"], "text": c["text"],
              "part": c["part"], "n_parts": c["n_parts"]} for c in chunks]
    (OUT / "chunks.json").write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")

    for name, path in [("base", cfg["base_model"]), ("finetuned", str(ROOT / cfg["output_dir"]))]:
        print(f"encoding corpus with {name} ...")
        d = Dense(path, cfg["query_prefix"], cfg["passage_prefix"])
        emb = d.encode_corpus(texts).astype(np.float16)   # half precision halves the shipped size
        np.save(OUT / f"emb_{name}.npy", emb)
        print(f"  saved {emb.shape} -> {OUT / f'emb_{name}.npy'}")
        del d

    print(f"\n{len(table)} chunks ready in {OUT}")


if __name__ == "__main__":
    main()
