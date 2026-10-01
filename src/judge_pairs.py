"""Phase 3.3b: an LLM judge decides whether each train/val chunk actually answers its question.

Usage:
    python src/judge_pairs.py --calibrate   # check the judge first (manual labels + wrong-pair controls)
    python src/judge_pairs.py               # judge all train/val candidates (resumable)

The judge chain uses different, stronger models than the question generators, so no model grades
its own output. Safe to stop and rerun; judged pairs are kept in data/synthetic/judgments.jsonl.
When every candidate is judged, writes data/synthetic/queries_filtered.jsonl:
  train/val pairs with verdict "yes" + test drafts (for human review, not judged).
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import yaml

from common import ROOT, load_env, read_jsonl
from llm import DailyQuotaExceeded, LLMChain, api_keys

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # show progress when output is redirected

SCORED = ROOT / "data" / "synthetic" / "queries_scored.jsonl"
JUDGMENTS = ROOT / "data" / "synthetic" / "judgments.jsonl"
KEPT = ROOT / "data" / "synthetic" / "queries_filtered.jsonl"
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
MANUAL = ROOT / "data" / "eval" / "judge_calibration_manual.jsonl"
RESULTS = ROOT / "results" / "phase3_filter"
STATE = ROOT / ".cache" / "llm_model_state.json"   # daily-quota benches, shared across scripts
VERDICTS = ["yes", "partial", "no"]

SYSTEM_PROMPT = """너는 한국 노동법 검색 데이터의 품질 검수자다. 각 항목은 일반인의 질문 하나와 법령 조문 또는 판례의 조각 하나다. 조각만 보고 판정한다.

판정 기준:
- "yes": 조각이 질문의 핵심 쟁점을 직접 다루고, 조각 내용만으로 질문의 핵심 답(가능 여부, 기간, 금액, 요건, 판단 기준 등)을 줄 수 있다. 답이 완전하지 않아도 핵심이 조각에 있으면 yes.
- "partial": 조각이 같은 주제와 관련은 있지만, 질문의 핵심 답은 이 조각이 아니라 다른 조문이나 판례에 있다.
- "no": 조각이 질문에 답하지 못하거나, 질문이 너무 막연해서 특정한 답이 없다.

일상어와 법률 용어가 달라도 뜻이 같으면 같은 것으로 본다(예: 알바비=임금, 잘렸다=해고, 빨간날=휴일).
reason은 한국어 한 문장으로 짧게 쓴다."""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "results": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "verdict": {"type": "STRING", "enum": VERDICTS},
                    "reason": {"type": "STRING"},
                },
                "required": ["id", "verdict", "reason"],
            },
        }
    },
    "required": ["results"],
}


def validate(parsed):
    for r in parsed["results"]:
        if r["verdict"] not in VERDICTS:
            raise ValueError(r["verdict"])


def judge_batch(chain, items, chunk_text, seed):
    """items: list of dicts with question + chunk_id. Returns ({index: (verdict, reason)}, model, version, seed)."""
    prompt = "각 항목을 판정하라. 결과의 id는 항목 id를 그대로 쓴다.\n\n" + "\n\n".join(
        f"[항목 id={i}]\n질문: {it['question']}\n조각:\n{chunk_text[it['chunk_id']]}" for i, it in enumerate(items))
    out = chain.generate_json(SYSTEM_PROMPT, prompt, SCHEMA, seed, validate=validate)
    if out is None:
        return None
    parsed, model, version, used_seed = out
    res = {}
    for r in parsed["results"]:
        try:
            res[int(str(r["id"]).strip())] = (r["verdict"], r["reason"])
        except ValueError:
            continue
    return res, model, version, used_seed


def calibrate(cfg, chain, qs_by_id, chunks, chunk_text, model=None):
    manual = read_jsonl(MANUAL)
    items = [{**qs_by_id[m["qid"]], "expected": m["label"], "kind": "manual"} for m in manual]
    # Wrong-pair controls: real questions paired with a chunk from a different law/case group.
    rng = random.Random(cfg["seed"])
    source = {c["chunk_id"]: c["metadata"].get("law_name", "precedent") for c in chunks}
    pool = [q for q in qs_by_id.values() if not q["drop_reason"] and q["split"] in cfg["judge"]["apply_to"]]
    for q in rng.sample(pool, 10):
        # A chunk from a different law (or law vs precedent), so it cannot legitimately answer.
        other = rng.choice([c for c in chunks if source[c["chunk_id"]] != source[q["chunk_id"]]])
        items.append({**q, "chunk_id": other["chunk_id"], "expected": "no", "kind": "wrong_pair"})
    size = cfg["judge"]["pairs_per_request"]
    results = []
    for b in range(0, len(items), size):
        batch = items[b:b + size]
        out = judge_batch(chain, batch, chunk_text, cfg["seed"] + b)
        if out is None:
            sys.exit("Judge failed on a calibration batch; rerun later.")
        res, model, _, _ = out
        for i, it in enumerate(batch):
            verdict, reason = res.get(i, (None, "missing"))
            results.append({"qid": it["qid"], "kind": it["kind"], "expected": it["expected"],
                            "verdict": verdict, "reason": reason, "judge_model": model,
                            "question": it["question"], "chunk_id": it["chunk_id"]})
    agree = sum(r["verdict"] == r["expected"] for r in results)
    # "keep" decision agreement is what matters for filtering: yes vs not-yes.
    keep_agree = sum((r["verdict"] == "yes") == (r["expected"] == "yes") for r in results)
    summary = {
        "n": len(results),
        "exact_agreement": round(agree / len(results), 4),
        "keep_decision_agreement": round(keep_agree / len(results), 4),
        "by_kind": {k: dict(Counter((r["expected"], r["verdict"]) for r in results if r["kind"] == k).most_common())
                    for k in ("manual", "wrong_pair")},
        "judge_models": dict(Counter(r["judge_model"] for r in results)),
        "caveat": "Manual labels are the assistant's own reading, not an independent expert.",
    }
    for k in summary["by_kind"]:
        summary["by_kind"][k] = {f"{e}->{v}": n for (e, v), n in Counter(
            (r["expected"], r["verdict"]) for r in results if r["kind"] == k).items()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = "judge_calibration.json" if not model else f"judge_calibration_{model.replace(':', '_').replace('/', '_')}.json"
    (RESULTS / name).write_text(
        json.dumps({"summary": summary, "items": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nDisagreements:")
    for r in results:
        if r["verdict"] != r["expected"]:
            print(f"  [{r['kind']}] expected {r['expected']}, judge {r['verdict']}: {r['question'][:60]} | {r['reason']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--finalize", action="store_true",
                    help="write queries_filtered.jsonl from the judgments made so far, without "
                         "judging the rest (unjudged pairs are kept and flagged judge_verdict=null)")
    ap.add_argument("--model", default=None, help="with --calibrate: test only this judge model")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "filter.yaml").read_text(encoding="utf-8"))
    jcfg = cfg["judge"]
    bad = [v for v in jcfg["keep_verdicts"] if v not in VERDICTS]
    if bad:   # unquoted "yes" in YAML becomes boolean True and silently drops every judged pair
        sys.exit(f"configs/filter.yaml judge.keep_verdicts has non-verdict values {bad}; "
                 f"quote them, e.g. keep_verdicts: [\"yes\"]")
    if args.model:
        jcfg = dict(jcfg, models=[args.model])
    keys = api_keys(load_env())
    chain = LLMChain(jcfg, keys, state_path=STATE)
    qs = read_jsonl(SCORED)
    qs_by_id = {q["qid"]: q for q in qs}
    chunks = read_jsonl(CHUNKS)
    chunk_text = {c["chunk_id"]: c["text"] for c in chunks}

    if args.calibrate:
        return calibrate(cfg, chain, qs_by_id, chunks, chunk_text, args.model)

    candidates = [q for q in qs if not q["drop_reason"] and q["split"] in jcfg["apply_to"]]
    done = {j["qid"] for j in read_jsonl(JUDGMENTS)} if JUDGMENTS.exists() else set()
    todo = [] if args.finalize else [q for q in candidates if q["qid"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Candidates: {len(candidates)}. Already judged: {len(done)}. To judge now: {len(todo)}")

    size = jcfg["pairs_per_request"]
    batches = [todo[i:i + size] for i in range(0, len(todo), size)]
    n = 0
    with JUDGMENTS.open("a", encoding="utf-8") as fj:
        try:
            for b, batch in enumerate(batches, 1):
                out = judge_batch(chain, batch, chunk_text, cfg["seed"] + 100000 + b * 1000)
                if out is None:
                    print(f"[{b}/{len(batches)}] batch failed; will retry on next run")
                    continue
                res, model, version, seed = out
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                got = Counter()
                for i, q in enumerate(batch):
                    if i not in res:
                        continue
                    verdict, reason = res[i]
                    fj.write(json.dumps({"qid": q["qid"], "verdict": verdict, "reason": reason,
                                         "judge_model": model, "judge_version": version, "seed": seed,
                                         "created_at": now}, ensure_ascii=False) + "\n")
                    got[verdict] += 1
                    n += 1
                fj.flush()
                print(f"[{b}/{len(batches)}] via {model}: {dict(got)}")
        except DailyQuotaExceeded as e:
            print(f"\nDaily free quota used up on every judge model ({e}). Progress saved; rerun tomorrow.")

    judgments = {j["qid"]: j for j in read_jsonl(JUDGMENTS)} if JUDGMENTS.exists() else {}
    remaining = [q for q in candidates if q["qid"] not in judgments]
    print(f"\nThis run: {n} judged. Total judged: {len(candidates) - len(remaining)}/{len(candidates)}")
    if remaining and not args.finalize:
        print("Not finished yet; rerun the same command to continue, "
              "or run with --finalize to build the dataset from what is judged so far.")
        return

    kept = []
    stats = defaultdict(Counter)
    for q in qs:
        if q["drop_reason"]:
            continue
        if q["split"] in jcfg["apply_to"]:
            j = judgments.get(q["qid"])
            if j is None:   # --finalize: not reached by the judge, kept but marked unverified
                q["judge_verdict"] = None
                stats["verdict_by_split"][f"{q['split']}:unjudged"] += 1
                kept.append(q)
                stats["kept_by_split"][q["split"]] += 1
                continue
            q["judge_verdict"], q["judge_reason"], q["judge_model"] = j["verdict"], j["reason"], j["judge_model"]
            stats["verdict_by_split"][f"{q['split']}:{j['verdict']}"] += 1
            stats["verdict_by_doc_type"][f"{q['doc_type']}:{j['verdict']}"] += 1
            stats["verdict_by_round_trip"][f"rt_{'pass' if q['round_trip_pass'] else 'fail'}:{j['verdict']}"] += 1
            stats["judge_models"][j["judge_model"]] += 1
            if j["verdict"] not in jcfg["keep_verdicts"]:
                continue
        kept.append(q)
        stats["kept_by_split"][q["split"]] += 1
    with KEPT.open("w", encoding="utf-8") as f:
        for q in kept:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    summary = {k: dict(v) for k, v in stats.items()}
    (RESULTS / "judge_stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(kept)} pairs to {KEPT}")


if __name__ == "__main__":
    main()
