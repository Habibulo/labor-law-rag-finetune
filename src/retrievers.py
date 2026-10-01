"""Shared retrievers: BM25 over Kiwi morphemes, and dense embedding search (sentence-transformers).

Both return a full score matrix [n_queries, n_chunks] so callers can compute ranks for any chunk.
"""
import hashlib
from pathlib import Path

import numpy as np

from common import ROOT

CACHE = ROOT / ".cache" / "embeddings"

# Content-bearing Kiwi tags: nouns, verb/adjective stems, roots, adverbs, foreign/number/hanja.
KEEP_TAGS = {"NNG", "NNP", "NR", "NP", "VV", "VA", "XR", "MAG", "SL", "SN", "SH"}


class BM25Kiwi:
    def __init__(self, texts):
        from kiwipiepy import Kiwi
        from rank_bm25 import BM25Okapi
        self.kiwi = Kiwi()
        self.bm25 = BM25Okapi([self.tokenize(t) for t in texts])

    def tokenize(self, text):
        return [f"{t.form}/{t.tag[:2]}" for t in self.kiwi.tokenize(text) if t.tag in KEEP_TAGS]

    def scores(self, queries):
        return np.stack([self.bm25.get_scores(self.tokenize(q)) for q in queries]).astype(np.float32)


class Dense:
    def __init__(self, name, query_prefix="", passage_prefix="", max_seq_length=512, batch_size=32):
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(name, device=device)
        if device == "cuda":
            self.model.half()
        self.model.max_seq_length = max_seq_length
        self.name, self.qp, self.pp, self.bs = name, query_prefix, passage_prefix, batch_size

    def encode(self, texts, prefix):
        return self.model.encode([prefix + t for t in texts], batch_size=self.bs, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=True).astype(np.float32)

    def encode_corpus(self, texts):
        """Corpus embeddings are cached on disk, keyed by model name + exact corpus text."""
        h = hashlib.sha1((self.name + self.pp + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
        f = CACHE / f"{self.name.replace('/', '__')}_{h}.npy"
        if f.exists():
            return np.load(f)
        emb = self.encode(texts, self.pp)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, emb)
        return emb

    def scores(self, queries, corpus_emb):
        return self.encode(queries, self.qp) @ corpus_emb.T


def group_ranks(scores, chunk_groups, target_groups):
    """1-based rank of the best-ranked chunk belonging to each query's target group."""
    order = np.argsort(-scores, axis=1)
    groups = np.asarray(chunk_groups)[order]
    hits = groups == np.asarray(target_groups)[:, None]
    return hits.argmax(axis=1) + 1
