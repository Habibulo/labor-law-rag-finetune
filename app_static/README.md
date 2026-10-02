---
title: 근로 Q&A 검색 — 파인튜닝 전/후 비교
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
---

Side-by-side Korean labor-law retrieval: an off-the-shelf embedding model versus one fine-tuned
on Korean labor law. Results are **precomputed** (both models were run offline over the same
3,922-chunk corpus) so the page is free to host and loads instantly.

Recall@1 0.513 → 0.657 (+28.1% relative) on 335 held-out questions.

Code, full evaluation and leakage audit: https://github.com/Habibulo/labor-law-rag-finetune
