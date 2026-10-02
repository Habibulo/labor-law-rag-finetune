"""Build a self-contained static demo page (free Hugging Face static Space / GitHub Pages).

Usage:
    python app_static/build.py

Gradio Spaces now need a PRO subscription, so the live demo falls back to the precomputed
option the project brief allowed. Both models are run here, offline, over a curated question
list; the page ships the results as embedded JSON and needs no server.

Questions come from two places, and the page labels which is which:
  - showcase: hand-written everyday questions (no ground truth known)
  - test set: held-out questions WITH their correct article, so a viewer sees who got it right

Output: app_static/index.html (self-contained), app_static/README.md (Space header)
"""
import json
import random
import sys

import numpy as np
import yaml

from common import ROOT, read_jsonl
from retrievers import Dense

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

OUT = ROOT / "app_static"
TOP_K = 3

SHOWCASE = [
    "주 15시간 일하는 알바인데 주휴수당 받을 수 있나요?",
    "5인 미만 사업장도 연차휴가가 있나요?",
    "작년에 안 준 알바비 지금이라도 받을 수 있나요?",
    "퇴근길에 슈퍼 들렀다가 사고 났는데 산재 처리 되나요?",
    "입사한 지 5개월인데 육아휴직 신청하면 회사가 거절할 수 있나요?",
    "사장님이 갑자기 내일부터 나오지 말라는데 수당 받을 수 있나요?",
    "수습기간에는 최저임금보다 적게 줘도 되나요?",
    "직장 상사가 계속 모욕적인 말을 하는데 법적으로 괴롭힘인가요?",
    "회사가 월급에서 지각비를 마음대로 떼어가는데 괜찮은 건가요?",
    "퇴사한 지 2주가 넘었는데 아직 퇴직금을 안 줍니다.",
]


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "train_embedding.yaml").read_text(encoding="utf-8"))
    chunks = read_jsonl(ROOT / "data" / "processed" / "chunks.jsonl")
    texts = [c["text"] for c in chunks]
    by_group = {}
    for c in chunks:
        by_group.setdefault(c["group_id"], []).append(c["chunk_id"])

    test_qs = [q for q in read_jsonl(ROOT / "data" / "synthetic" / "queries_filtered.jsonl")
               if q["split"] == "test"]
    # 120 rather than a handful: a 40-question draw gave 18 vs 21 top-1 hits, well below the
    # full-set 0.513 vs 0.657, purely from sampling noise. The page states the sample size.
    rng = random.Random(7)
    sample = rng.sample(test_qs, 120)

    questions = [{"q": q, "kind": "showcase", "gold": None} for q in SHOWCASE]
    questions += [{"q": t["question"], "kind": "test", "gold": t["group_id"]} for t in sample]

    models = {}
    for name, path in [("base", cfg["base_model"]), ("finetuned", str(ROOT / cfg["output_dir"]))]:
        print(f"encoding with {name} ...")
        d = Dense(path, cfg["query_prefix"], cfg["passage_prefix"])
        corpus = d.encode_corpus(texts)
        qemb = d.encode([q["q"] for q in questions], cfg["query_prefix"])
        models[name] = qemb @ corpus.T
        del d

    rows = []
    for i, item in enumerate(questions):
        entry = {"q": item["q"], "kind": item["kind"], "gold": None, "results": {}}
        gold_ids = set()
        if item["gold"]:
            gold_ids = set(by_group[item["gold"]])
            gold_chunk = next(c for c in chunks if c["chunk_id"] in gold_ids)
            entry["gold"] = {"title": gold_chunk["title"], "ids": sorted(gold_ids)}
        for name in ("base", "finetuned"):
            top = np.argsort(-models[name][i])[:TOP_K]
            entry["results"][name] = [{
                "title": chunks[j]["title"],
                "type": chunks[j]["doc_type"],
                "snippet": chunks[j]["text"].split("\n", 1)[-1][:220],
                "correct": bool(item["gold"] and chunks[j]["chunk_id"] in gold_ids),
            } for j in top]
        rows.append(entry)

    metrics = json.loads((ROOT / "results/phase4_baselines/metrics.json").read_text(encoding="utf-8"))
    payload = {"questions": rows,
               "metrics": {k: {m: v[m] for m in ("recall@1", "recall@10", "ndcg@10")}
                           for k, v in metrics.items() if not k.startswith("_")},
               "n_queries": metrics["_meta"]["n_queries"], "n_chunks": metrics["_meta"]["n_chunks"]}

    hits = {n: sum(r["results"][n][0]["correct"] for r in rows if r["kind"] == "test")
            for n in ("base", "finetuned")}
    n_test = sum(1 for r in rows if r["kind"] == "test")
    print(f"\nsampled test questions: {n_test} | top-1 correct — base {hits['base']}, "
          f"fine-tuned {hits['finetuned']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False)),
                                    encoding="utf-8")
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")
    print(f"wrote {OUT / 'index.html'}")


SPACE_README = """---
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
"""

HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>근로 Q&A 검색 — 파인튜닝 전/후 비교</title>
<style>
 :root{--bg:#f6f8fa;--card:#fff;--line:#d0d7de;--mut:#57606a;--acc:#0969da;--ok:#1a7f37;--bad:#cf222e;
       color-scheme:light}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:#1f2328;font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
 .wrap{max-width:1100px;margin:0 auto;padding:24px 16px 60px}
 h1{font-size:24px;margin:0 0 6px} h2{font-size:15px;margin:0 0 8px}
 .sub{color:var(--mut);margin-bottom:16px}
 .banner{background:#fff8c5;border:1px solid #d4a72c;border-radius:8px;padding:10px 14px;margin-bottom:18px;font-size:14px}
 .stats{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:18px}
 .stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;flex:1;min-width:150px}
 .stat b{display:block;font-size:20px} .stat span{color:var(--mut);font-size:12px}
 .ctl{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}
 input,select{padding:9px 12px;border:1px solid var(--line);border-radius:8px;font-size:14px;background:#fff}
 input{flex:1;min-width:220px}
 .q{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
 .qt{font-weight:600;margin-bottom:4px}
 .gold{font-size:13px;color:var(--mut);margin-bottom:10px}
 .gold b{color:var(--ok)}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:12px}
 @media(max-width:760px){.cols{grid-template-columns:1fr}}
 .col h3{font-size:13px;color:var(--mut);margin:0 0 6px;text-transform:uppercase;letter-spacing:.04em}
 .hit{border:1px solid var(--line);border-left:3px solid var(--line);border-radius:6px;padding:8px 10px;margin-bottom:6px;background:#fff}
 .hit.correct{border-left-color:var(--ok);background:#f0fff4}
 .hit .t{font-size:13px;font-weight:600;color:var(--acc)}
 .hit .s{font-size:12px;color:#444;margin-top:3px}
 .tag{font-size:11px;color:var(--mut)}
 .mark{float:right;font-size:12px;font-weight:700}
 .mark.y{color:var(--ok)} .mark.n{color:var(--bad)}
 .pill{display:inline-block;font-size:11px;padding:1px 7px;border-radius:999px;background:#eaeef2;color:var(--mut);margin-left:6px}
 footer{margin-top:28px;color:var(--mut);font-size:13px}
 a{color:var(--acc)}
 table{border-collapse:collapse;width:100%;font-size:13px;background:var(--card)}
 th,td{border:1px solid var(--line);padding:6px 9px;text-align:left}
 th{background:#eaeef2}
</style></head><body><div class="wrap">
<div class="banner"><b>⚠️ 포트폴리오용 연구 프로젝트이며 법률 자문이 아닙니다.</b><br>
Portfolio research project, not legal advice. 법령 스냅샷 기준일: 2026-09-25.</div>

<h1>근로 Q&A 도메인 적응 — 검색 결과 비교</h1>
<div class="sub">일상 한국어 노동 질문에 대해 <b>기본 임베딩 모델</b>과 <b>노동법으로 파인튜닝한 모델</b>이
각각 찾아온 법령·판례입니다. 두 모델을 동일한 3,922개 청크 말뭉치에서 오프라인으로 실행해 미리 계산한 결과입니다.</div>

<div class="stats" id="stats"></div>
<div id="table"></div>

<div class="ctl" style="margin-top:18px">
 <input id="search" placeholder="질문 검색…">
 <select id="filter">
   <option value="all">전체 질문</option>
   <option value="test">정답이 있는 평가셋 질문</option>
   <option value="showcase">예시 질문</option>
   <option value="diff">두 모델의 1순위가 다른 질문</option>
 </select>
</div>
<div id="list"></div>

<footer>데이터 출처: 법제처 국가법령정보 OpenAPI (via
<a href="https://github.com/legalize-kr/legalize-kr">legalize-kr</a>). 법령·판례 원문은 저작권법 제7조에 따라
보호받지 못하는 저작물입니다. ·
<a href="https://github.com/Habibulo/labor-law-rag-finetune">코드 및 전체 평가 결과</a></footer>
</div>
<script>
const DATA = __DATA__;
const TYPE = {law:"법률", decree:"시행령", rule:"시행규칙", precedent:"판례"};
const esc = s => s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

const test = DATA.questions.filter(q => q.kind === "test");
const top1 = n => test.filter(q => q.results[n][0].correct).length;
document.getElementById("stats").innerHTML = [
  ["Recall@1 (기본)", DATA.metrics["multilingual-e5-base"]["recall@1"].toFixed(3), "검증 질문 " + DATA.n_queries + "건"],
  ["Recall@1 (파인튜닝)", DATA.metrics["e5-base FINE-TUNED"]["recall@1"].toFixed(3), "상대 +28.1%"],
  ["이 페이지 표본", top1("base") + " → " + top1("finetuned"),
   "무작위 표본 " + test.length + "건 중 1순위 정답 (전체 지표는 335건 기준)"],
  ["말뭉치", DATA.n_chunks.toLocaleString(), "청크 (법령 11종 + 판례 507건)"]
].map(([t,v,s]) => `<div class="stat"><span>${t}</span><b>${v}</b><span>${s}</span></div>`).join("");

document.getElementById("table").innerHTML = "<table><tr><th>모델</th><th>Recall@1</th><th>Recall@10</th><th>NDCG@10</th></tr>" +
  Object.entries(DATA.metrics).map(([k,v]) =>
    `<tr><td>${k === "e5-base FINE-TUNED" ? "<b>"+esc(k)+"</b>" : esc(k)}</td>` +
    `<td>${v["recall@1"].toFixed(3)}</td><td>${v["recall@10"].toFixed(3)}</td><td>${v["ndcg@10"].toFixed(3)}</td></tr>`
  ).join("") + "</table>";

function hit(h, showMark){
  const mark = showMark ? `<span class="mark ${h.correct?"y":"n"}">${h.correct?"정답":"✗"}</span>` : "";
  return `<div class="hit ${h.correct?"correct":""}">${mark}
    <div class="t">${esc(h.title)}</div>
    <div class="tag">${TYPE[h.type]||h.type}</div>
    <div class="s">${esc(h.snippet)} …</div></div>`;
}
function card(q){
  const showMark = q.kind === "test";
  const gold = q.gold ? `<div class="gold">정답 조문: <b>${esc(q.gold.title)}</b></div>` : "";
  return `<div class="q"><div class="qt">${esc(q.q)}
    <span class="pill">${q.kind === "test" ? "평가셋" : "예시"}</span></div>${gold}
    <div class="cols">
      <div class="col"><h3>기본 모델</h3>${q.results.base.map(h=>hit(h,showMark)).join("")}</div>
      <div class="col"><h3>파인튜닝 모델</h3>${q.results.finetuned.map(h=>hit(h,showMark)).join("")}</div>
    </div></div>`;
}
function render(){
  const term = document.getElementById("search").value.trim();
  const f = document.getElementById("filter").value;
  let rows = DATA.questions;
  if (f === "test" || f === "showcase") rows = rows.filter(q => q.kind === f);
  if (f === "diff") rows = rows.filter(q => q.results.base[0].title !== q.results.finetuned[0].title);
  if (term) rows = rows.filter(q => q.q.includes(term));
  document.getElementById("list").innerHTML =
    rows.length ? rows.map(card).join("") : "<div class='q'>해당하는 질문이 없습니다.</div>";
}
document.getElementById("search").addEventListener("input", render);
document.getElementById("filter").addEventListener("change", render);
render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
