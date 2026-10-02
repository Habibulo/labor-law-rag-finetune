---
title: 근로 Q&A 검색 — 파인튜닝 전/후 비교
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.0
app_file: app.py
pinned: false
license: mit
---

# 근로 Q&A 도메인 적응 — Korean Labor-Law Retrieval

Type an everyday Korean question about work and see, side by side, what an **off-the-shelf**
Korean embedding model retrieves versus one **fine-tuned on Korean labor law**.

**Corpus:** 11 statutes with their decrees and rules + 507 Supreme Court rulings → 3,922 chunks.
**Result on 335 held-out questions:** Recall@1 **0.513 → 0.657** (+28.1% relative), NDCG@10 0.711 → 0.815.

> ⚠️ Portfolio research project, not legal advice. 법률 자문이 아닙니다. Law snapshot: 2026-09-25.

Code and full evaluation: https://github.com/Habibulo/labor-law-rag-finetune

## How it runs on free CPU

Corpus embeddings for both models are precomputed (`precompute.py`) and shipped as `assets/*.npy`,
so only the question is encoded at request time.

## Deploy

```bash
# the contents of this folder become the Space repo
hf upload <user>/<space-name> . . --repo-type=space
```

Set `FINETUNED_MODEL` to the Hub id of the fine-tuned model (defaults to
`Habibulo/e5-base-labor-law-ko`).
