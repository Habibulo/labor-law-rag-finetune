"""Phase 3.2: generate everyday-Korean questions for each chunk with the Gemini API (free tier).

Usage:
    python src/generate_queries.py                  # all splits: test first, then val, then train
    python src/generate_queries.py --limit 10       # quick trial on 10 chunks
    python src/generate_queries.py --splits test

Needs GEMINI_API_KEY in the project .env file. Safe to stop and rerun at any time: finished
chunks are recorded in data/synthetic/generation_log.jsonl and skipped on the next run.
When the daily free quota runs out the script stops cleanly; rerun it the next day.

Output: data/synthetic/queries_raw.jsonl (one question per line), generation_log.jsonl
"""
import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone

import yaml

from common import ROOT, load_env, read_jsonl
from llm import DailyQuotaExceeded, LLMChain, api_keys

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # show progress when output is redirected

CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
SPLITS = ROOT / "data" / "processed" / "splits.json"
OUT_DIR = ROOT / "data" / "synthetic"
QUERIES = OUT_DIR / "queries_raw.jsonl"
LOG = OUT_DIR / "generation_log.jsonl"
FAILURES = OUT_DIR / "generation_failures.jsonl"

SYSTEM_PROMPT = """너는 한국 노동법 검색 시스템의 학습 데이터를 만드는 사람이다.
각 법령 조문 또는 판례 조각을 읽고, 법을 모르는 일반인이 인터넷 커뮤니티나 노동 상담 창구에 실제로 물어볼 법한 질문을 만든다. 질문의 답은 반드시 그 조각 안에 있어야 한다.

규칙:
1. 질문자는 다양하게: 알바생, 정규직 직원, 계약직·파견직 근로자, 사장님(소규모 사업주), 출산·육아휴직 예정자, 퇴사 예정자, 일하다 다친 근로자, 인사담당자.
2. 자연스러운 구어체로 쓴다. 절반 이상은 상황 설명형("~한 상황인데 ~해도 되나요?"), 나머지는 짧은 궁금증형.
3. 조각에 나오는 법률 용어는 일반인이 실제로 쓰는 말로 바꾼다. 예: 소정근로시간→원래 일하기로 한 시간, 사용자→사장님/회사, 해고의 예고→미리 잘린다고 알려주는 것, 통상임금→평소 받는 월급. 단, 주휴수당·연차·퇴직금·최저임금·육아휴직·실업급여·산재처럼 일반인도 흔히 쓰는 말은 그대로 써도 된다.
4. 법령 이름, 조문 번호(제○조), 사건번호, 조각의 문장을 그대로 베끼지 않는다.
5. 질문 하나만 읽어도 무엇을 묻는지 알 수 있게 구체적으로 쓴다. "이거 불법인가요?" 같은 모호한 질문은 금지.
6. 위원회 구성, 행정 서식, 기관 내부 절차처럼 일반인이나 사업주가 실제로 물어볼 일이 거의 없는 조각은 질문 수를 줄이거나 0개로 한다. 억지로 만들지 않는다.
7. 한 조각의 질문들은 서로 다른 내용과 다른 질문자를 다룬다."""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "results": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "questions": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "persona": {"type": "STRING"},
                                "type": {"type": "STRING", "enum": ["situational", "definitional"]},
                                "question": {"type": "STRING"},
                            },
                            "required": ["persona", "type", "question"],
                        },
                    },
                },
                "required": ["id", "questions"],
            },
        }
    },
    "required": ["results"],
}

ARTICLE_NO_RE = re.compile(r"제\s*\d+\s*조")
# Formal law names and their common short forms. Everyday words like 노동법/불법/방법 are not flagged.
LAW_NAME_RE = re.compile(
    r"근로기준법|최저임금법|퇴직급여\s*보장법|남녀고용평등|기간제법|파견법|고용보험법|산재보험법|"
    r"산업재해보상보험법|임금채권보장법|근로자참여|보험료징수|징수법|시행령|시행규칙")



def _validate(parsed):
    for r in parsed["results"]:
        str(r["id"]), r["questions"]


def generate(batch, split, chain, gcfg, base_seed):
    """Return (results id->questions, model, modelVersion, seed), or None if every attempt failed."""
    n = gcfg["max_questions"][split]
    prompt = f"각 조각마다 질문을 0~{n}개 만들어라. 결과의 id는 조각 id를 그대로 쓴다.\n\n" + "\n\n".join(
        f"[조각 id={i}]\n{c['text']}" for i, c in enumerate(batch))
    out = chain.generate_json(SYSTEM_PROMPT, prompt, RESPONSE_SCHEMA, base_seed, validate=_validate)
    if out is None:
        return None
    parsed, model, version, seed = out
    return {str(r["id"]).strip(): r["questions"] for r in parsed["results"]}, model, version, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process at most this many new chunks")
    ap.add_argument("--splits", default="test,val,train")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "queries.yaml").read_text(encoding="utf-8"))
    gcfg = cfg["generation"]
    keys = api_keys(load_env())
    if "GEMINI_API_KEY" not in keys:
        sys.exit("GEMINI_API_KEY missing from .env")
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    chunks = read_jsonl(CHUNKS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = {r["chunk_id"] for r in read_jsonl(LOG)} if LOG.exists() else set()

    todo = []
    for split in args.splits.split(","):
        todo += [(split, c) for c in chunks if splits[c["group_id"]] == split and c["chunk_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Already done: {len(done)} chunks. To do now: {len(todo)}")

    size = gcfg["chunks_per_request"]
    batches = []
    for split in args.splits.split(","):
        items = [c for s, c in todo if s == split]
        batches += [(split, items[i:i + size]) for i in range(0, len(items), size)]

    def write(batch, split, result, fq, fl):
        results, model, version, seed = result
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        added = 0
        for i, c in enumerate(batch):
            qs = results.get(str(i))
            if qs is None:
                print(f"  missing result for {c['chunk_id']}; will retry on next run")
                continue
            for q in qs:
                q["question"] = unicodedata.normalize("NFC", q.get("question", "")).strip()
            kept = [q for q in qs if q["question"]][: gcfg["max_questions"][split]]
            for k, q in enumerate(kept):
                fq.write(json.dumps({
                    "qid": f"{c['chunk_id']}|q{k}",
                    "question": q["question"],
                    "persona": q.get("persona", ""),
                    "qtype": q.get("type", ""),
                    "chunk_id": c["chunk_id"],
                    "group_id": c["group_id"],
                    "doc_type": c["doc_type"],
                    "split": split,
                    "mentions_article_no": bool(ARTICLE_NO_RE.search(q["question"])),
                    "mentions_law_name": bool(LAW_NAME_RE.search(q["question"])),
                    "model": model,
                    "model_version": version,
                    "seed": seed,
                    "prompt_version": gcfg["prompt_version"],
                    "created_at": now,
                }, ensure_ascii=False) + "\n")
            fl.write(json.dumps({"chunk_id": c["chunk_id"], "split": split, "n_questions": len(kept),
                                 "model": model, "seed": seed, "created_at": now}, ensure_ascii=False) + "\n")
            added += len(kept)
        fq.flush()
        fl.flush()
        return added, model

    n_questions, failed = 0, []
    chain = LLMChain(gcfg, keys, state_path=ROOT / ".cache" / "llm_model_state.json")

    with QUERIES.open("a", encoding="utf-8") as fq, LOG.open("a", encoding="utf-8") as fl:
        try:
            for b, (split, batch) in enumerate(batches, 1):
                result = generate(batch, split, chain, gcfg, cfg["seed"] + b * 1000)
                if result:
                    added, model = write(batch, split, result, fq, fl)
                    n_questions += added
                    print(f"[{b}/{len(batches)}] {split} via {model}: +{added} questions")
                    continue
                # Whole batch failed: isolate the problem by retrying chunk by chunk.
                print(f"[{b}/{len(batches)}] batch failed; retrying its {len(batch)} chunks one by one")
                for j, c in enumerate(batch):
                    result = generate([c], split, chain, gcfg, cfg["seed"] + b * 1000 + j + 1)
                    if result:
                        added, model = write([c], split, result, fq, fl)
                        n_questions += added
                    else:
                        failed.append(c["chunk_id"])
                        with FAILURES.open("a", encoding="utf-8") as ff:
                            ff.write(json.dumps({"chunk_id": c["chunk_id"], "split": split,
                                                 "at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                                                ensure_ascii=False) + "\n")
                        print(f"  gave up on {c['chunk_id']} for this run (logged; rerun retries it)")
        except DailyQuotaExceeded as e:
            print(f"\nDaily free quota used up on every model ({e}). Progress is saved; rerun tomorrow.")

    total_done = len({r["chunk_id"] for r in read_jsonl(LOG)}) if LOG.exists() else 0
    print(f"\nThis run: {n_questions} questions, {len(failed)} chunks failed. "
          f"Chunks done overall: {total_done}/{len(chunks)}")


if __name__ == "__main__":
    main()
