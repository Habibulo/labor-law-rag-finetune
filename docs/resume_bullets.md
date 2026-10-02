# Resume & portfolio material

Every number below comes from a script in this repo. Sources:
`results/phase4_baselines/metrics.json`, `results/phase5_embedding/train_meta.json`,
`results/phase4_baselines/leakage_check.json`, `results/phase3_filter/`.

---

## Resume bullets — English

**Project title:** Korean Labor-Law Retrieval — Domain-Adapted Embedding Fine-Tuning
*(Python, PyTorch, sentence-transformers, Hugging Face, BM25/Kiwi)*

**Links:** [demo](https://huggingface.co/spaces/Khabib1304/labor-law-retrieval) ·
[model](https://huggingface.co/Khabib1304/e5-base-labor-law-ko) ·
[code](https://github.com/Habibulo/labor-law-rag-finetune)

- Fine-tuned `multilingual-e5-base` on 3,631 domain question/passage pairs, raising **Recall@1 from
  0.513 to 0.657 (+28.1% relative)** and **NDCG@10 from 0.711 to 0.815** on 335 held-out queries,
  with non-overlapping 95% bootstrap confidence intervals.
- Built a reproducible Korean legal corpus pipeline — 11 statutes with decrees and rules plus 507
  Supreme Court rulings parsed into **2,543 structured documents and 3,922 retrieval chunks** —
  pinning every statute to the version legally *in force* on a fixed snapshot date by walking
  per-file git history.
- Generated **4,444 everyday-language training questions** using free-tier LLMs behind a
  5-model fallback client with per-provider quota detection, persistent cooldowns and resumable
  checkpointing, achieving zero-cost data generation after measuring each provider's real limits.
- Designed an **LLM-as-judge** data filter and validated it before use against 20 hand-labelled
  pairs and 10 deliberately mismatched controls; three independent judge models scored
  **90–93% agreement and rejected 10/10 controls**.
- Wrote a **leakage audit** that gates all reported metrics; it detected duplicate upstream court
  records placing identical text in two splits and excluded the affected queries, keeping the
  headline result defensible.
- Delivered the full result on **free compute only** — 3.2 GPU-minutes on a consumer RTX 5060
  (6.3 GB peak) and free API tiers — with pinned dependencies, YAML configs and fixed seeds.

### Short version (3 bullets, for a dense resume)

- Fine-tuned a Korean embedding model for labor-law retrieval, improving **Recall@1 by 28%**
  (0.513 → 0.657) and NDCG@10 by 15% on 335 held-out queries with non-overlapping 95% CIs.
- Built the end-to-end pipeline — statute/precedent ingestion, structure-aware chunking,
  LLM-generated training data with judge-based filtering — producing 3,922 chunks and 4,444 queries
  at **zero cost** on free APIs and one consumer GPU (3.2 min training).
- Added a **leakage audit** that caught duplicate upstream records contaminating the test split,
  and reported all metrics with bootstrap confidence intervals.

---

## 이력서 bullet — 한국어

**프로젝트:** 한국 노동법 검색을 위한 임베딩 모델 도메인 적응
*(Python, PyTorch, sentence-transformers, Hugging Face, BM25/Kiwi)*

- 노동법 도메인 질문–조문 쌍 3,631건으로 `multilingual-e5-base`를 파인튜닝하여, 검증용 질문 335건에서
  **Recall@1을 0.513 → 0.657 (상대 +28.1%)**, **NDCG@10을 0.711 → 0.815**로 향상 (95% 신뢰구간 비중첩).
- 법률 11종(시행령·시행규칙 포함)과 대법원 판례 507건을 **문서 2,543건 · 검색 청크 3,922건**으로 구조화하고,
  각 법령을 스냅샷 시점에 **실제 시행 중인 버전**으로 git 이력을 추적해 고정하는 재현 가능한 파이프라인 구축.
- 무료 API 5종 폴백 체인(제공자별 쿼터 판별·쿨다운 유지·재시작 가능 체크포인트)을 설계해
  **구어체 학습 질문 4,444건**을 비용 없이 생성.
- 학습 데이터 필터링용 **LLM-as-judge**를 설계하고, 수작업 라벨 20건과 오답 대조군 10건으로 사전 검증하여
  독립적인 판정 모델 3종에서 **일치율 90–93%, 대조군 10/10 기각** 확인.
- 모든 지표 산출 전에 실행되는 **데이터 누수 감사 스크립트**를 작성, 상위 저장소의 중복 판례로 인해
  동일 본문이 학습/평가 양쪽에 존재하던 문제를 탐지·제외하여 결과의 신뢰성 확보.
- 소비자용 GPU(RTX 5060, 최대 6.3GB) 3.2분 학습과 무료 API만으로 전 과정을 수행, 의존성 고정·YAML 설정·
  시드 고정으로 재현성 확보.

---

## 2-minute interview explanation — English

> Korean workers don't ask questions the way the law is written. Someone types
> "작년에 안 준 알바비 지금이라도 받을 수 있나요?" — can I still claim last year's part-time wages —
> and the statute says "임금채권은 3년간 행사하지 아니하면 시효로 소멸한다." No shared words.
> On that exact query, BM25 ranked the correct article 3,885th.
>
> So I built a retrieval system over Korean labor law and measured how much domain fine-tuning
> closes that gap. I collected 11 statutes with their decrees and rules, plus 507 Supreme Court
> rulings, and parsed them into the legal hierarchy — 장, 조, 항, 호, 목 — because an article is the
> unit a lawyer cites. One detail that mattered: the public mirror stores the latest *promulgated*
> text, not the text in force. 근로기준법's head revision doesn't take effect until 2027. So I walk
> each file's git history to the version actually in force on my snapshot date.
>
> For training data I had passages but no questions, so I generated 4,444 everyday-language
> questions with free LLM APIs, instructed to avoid legal vocabulary — that's what creates the gap
> I want to train on. Then I filtered them. My original plan was to drop any pair a baseline
> retriever couldn't already find, but when I inspected 20 rejected pairs, 19 were correct — they
> were exactly the hard vocabulary-gap examples. Keeping that filter would have deleted the signal.
> So I replaced it with an LLM judge that asks "does this passage actually answer this question?",
> and I calibrated the judge against my own labels plus deliberately mismatched controls before
> trusting it.
>
> The result: Recall@1 went from 0.513 to 0.657, about 28% relative, with non-overlapping
> confidence intervals. Training took 3.2 minutes on an RTX 5060. Two things surprised me. First,
> the fine-tuned dense model beat hybrid BM25+dense — once the vocabulary gap is closed, fusing
> with keyword search hurts. Second, my leakage audit failed: it found a court ruling duplicated
> upstream under two IDs, putting identical text in both splits. It was only 2 of 337 queries and
> excluding them barely moved the numbers, but I'd rather report that than discover it in an
> interview.
>
> What's not done: the test set is still synthetic, so the next step is human-verified questions,
> then the generation side — a small LLM that answers only from retrieved text, cites the article,
> and refuses when the answer isn't there.

## 2분 설명 — 한국어

> 한국의 근로자는 법전의 언어로 질문하지 않습니다. "작년에 안 준 알바비 지금이라도 받을 수 있나요?"라고
> 묻지만, 법은 "임금채권은 3년간 행사하지 아니하면 시효로 소멸한다"라고 되어 있습니다. 겹치는 단어가
> 없습니다. 실제로 이 질문에서 BM25는 정답 조문을 3,885위로 매겼습니다.
>
> 그래서 노동법 검색 시스템을 만들고, 도메인 파인튜닝이 이 격차를 얼마나 줄이는지 측정했습니다. 법률
> 11종과 시행령·시행규칙, 대법원 판례 507건을 수집해 장·조·항·호·목 구조로 파싱했습니다. 조문이 곧
> 인용 단위이기 때문입니다. 중요한 세부사항이 하나 있었는데, 공개 저장소는 최신 *공포* 본문을 담고
> 있지  실제 시행 중인 본문이 아니라는 점입니다. 근로기준법 최신 개정은 2027년에야 시행됩니다. 그래서
> 각 파일의 git 이력을 거슬러 올라가 스냅샷 시점에 시행 중이던 버전으로 고정했습니다.
>
> 학습 데이터는 지문만 있고 질문이 없어서, 무료 LLM API로 구어체 질문 4,444건을 생성했습니다. 법률
> 용어를 쓰지 말라고 지시했는데, 그것이 바로 학습해야 할 격차를 만들기 때문입니다. 원래 계획은 기존
> 검색기가 못 찾는 쌍을 버리는 것이었지만, 버려질 20건을 직접 확인해 보니 19건이 정답이었습니다.
> 바로 그 어려운 사례들이었죠. 그 필터를 유지했다면 학습 신호 자체를 지울 뻔했습니다. 대신 "이 지문이
> 실제로 이 질문에 답하는가"를 판정하는 LLM 심사자를 도입하고, 사용 전에 제 수작업 라벨과 오답
> 대조군으로 검증했습니다.
>
> 결과는 Recall@1 0.513 → 0.657, 상대 약 28% 향상이며 신뢰구간이 겹치지 않습니다. 학습은 RTX 5060에서
> 3.2분 걸렸습니다. 놀란 점이 둘 있었습니다. 첫째, 파인튜닝한 밀집 검색이 BM25 하이브리드보다 좋았습니다.
> 어휘 격차가 해소되자 키워드 검색과의 결합이 오히려 성능을 떨어뜨렸습니다. 둘째, 누수 감사가 실패했습니다.
> 상위 저장소에 같은 판례가 두 개의 일련번호로 중복 저장되어 동일 본문이 학습·평가 양쪽에 들어가 있었습니다.
> 337건 중 2건뿐이고 제외해도 수치는 거의 그대로였지만, 면접에서 지적받기 전에 제가 먼저 보고하는 편이 낫습니다.

---

## Likely interview questions

**Q: Why does domain fine-tuning help so much here?**
The base model was trained on general web text, where "알바비" and "임금채권" rarely co-occur. Labor
law has a narrow, consistent mapping between colloquial and statutory terms. Contrastive training
with in-batch negatives reshapes the space so a question lands near its own article and away from
the ~3,900 others. The gain is concentrated at Recall@1 (+14.4 points), which is where RAG quality
is decided.

**Q: How do you know your test set isn't biased by the query generator?**
I don't fully, and that's the top limitation in my README. The generator wrote both train and test
questions, so absolute numbers may be optimistic. What *is* sound is the comparison: every system
sees the identical queries and corpus. The planned fix is the 50 questions I write myself without
looking at the corpus, scored as a separate subset.

**Q: Why not just use hybrid BM25 + dense?**
I measured it. Hybrid RRF scored Recall@1 0.606 versus 0.657 for the fine-tuned model alone. BM25
was compensating for the vocabulary gap; once fine-tuning closes it, BM25's weaker ranking drags
the fusion down. Before fine-tuning, hybrid *was* best (0.552 vs 0.513).

**Q: How did you prevent leakage?**
Splits are by article/case, not by chunk, so all 항-splits of an article stay together. I also wrote
`check_leakage.py`, which runs before any number is reported and checks group overlap, chunk-id
overlap, identical text and near-duplicate questions. It caught a real problem: one ruling appears
twice upstream under different IDs. Those queries are excluded.

**Q: Why trust an LLM judge on legal data?**
I didn't trust it by default. I hand-labelled 20 pairs and added 10 pairs deliberately matched to
the wrong law. A judge that always says "yes" would score 95% on the hand-labels alone, so the
controls are the real test — all three judges rejected 10/10. They also disagreed with me on the
same 3 items, which pointed to genuine ambiguity rather than model error.

**Q: What would you do with more time or budget?**
In order: a human-verified test set; hard-negative mining (currently only in-batch negatives);
the generation track with citation and refusal; and a forgetting check on a general Korean
retrieval benchmark, since 2 epochs of narrow-domain training may have cost general performance —
I have not measured that.

**Q: Why are only 1,626 of 4,102 pairs judged?**
Free-tier quotas. I measured each provider's real ceiling and the judge rejected only 2.9% of what
it saw, so the expected yield from the remaining 2,476 pairs was ~70 bad pairs out of 3,631 — not
worth days of waiting. Unjudged pairs are kept and flagged `judge_verdict: null`, so the exact
composition is auditable.
