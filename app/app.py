"""Gradio demo: side-by-side labor-law retrieval, off-the-shelf vs fine-tuned.

Run locally:   python app/app.py
On Spaces:     this file is the entrypoint; set FINETUNED_MODEL to the Hub id.

Corpus embeddings are precomputed (app/precompute.py), so only the question is encoded at
request time and the free CPU tier stays responsive.
"""
import json
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
BASE_MODEL = os.environ.get("BASE_MODEL", "intfloat/multilingual-e5-base")
_LOCAL_FT = HERE.parent / "models" / "e5-base-labor-law"
FINETUNED_MODEL = os.environ.get(
    "FINETUNED_MODEL", str(_LOCAL_FT) if _LOCAL_FT.exists() else "Khabib1304/e5-base-labor-law-ko")
TOP_K = 5

CHUNKS = json.loads((ASSETS / "chunks.json").read_text(encoding="utf-8"))
EMB = {"base": np.load(ASSETS / "emb_base.npy").astype(np.float32),
       "finetuned": np.load(ASSETS / "emb_finetuned.npy").astype(np.float32)}
MODELS = {"base": SentenceTransformer(BASE_MODEL), "finetuned": SentenceTransformer(FINETUNED_MODEL)}
for m in MODELS.values():
    m.max_seq_length = 512

EXAMPLES = [
    "주 15시간 일하는 알바인데 주휴수당 받을 수 있나요?",
    "5인 미만 사업장도 연차휴가가 있나요?",
    "작년에 안 준 알바비 지금이라도 받을 수 있나요?",
    "퇴근길에 슈퍼 들렀다가 사고 났는데 산재 처리 되나요?",
    "입사한 지 5개월인데 육아휴직 신청하면 회사가 거절할 수 있나요?",
    "사장님이 갑자기 내일부터 나오지 말라는데 수당 받을 수 있나요?",
    "수습기간에는 최저임금보다 적게 줘도 되나요?",
    "직장 상사가 계속 모욕적인 말을 하는데 법적으로 괴롭힘인가요?",
]

TYPE_LABEL = {"law": "법률", "decree": "시행령", "rule": "시행규칙", "precedent": "판례"}


def search(question, which):
    q = MODELS[which].encode([f"query: {question}"], normalize_embeddings=True,
                             convert_to_numpy=True).astype(np.float32)
    scores = (q @ EMB[which].T)[0]
    top = np.argsort(-scores)[:TOP_K]
    return [(CHUNKS[i], float(scores[i])) for i in top]


def render(results):
    rows = []
    for rank, (c, score) in enumerate(results, 1):
        part = f" <i>(part {c['part']}/{c['n_parts']})</i>" if c["n_parts"] > 1 else ""
        body = c["text"].split("\n", 1)[-1][:260].replace("<", "&lt;")
        rows.append(
            f"<div style='border:1px solid #d0d7de;border-left:4px solid #0969da;border-radius:8px;"
            f"padding:10px 12px;margin-bottom:8px;background:#fff'>"
            f"<div style='font-size:12px;color:#57606a'>#{rank} · {TYPE_LABEL.get(c['doc_type'], c['doc_type'])}"
            f" · 유사도 {score:.3f}{part}</div>"
            f"<div style='font-weight:600;margin:4px 0;color:#0969da'>{c['title']}</div>"
            f"<div style='font-size:13px;color:#24292f;line-height:1.5'>{body} …</div></div>")
    return "".join(rows)


def compare(question):
    if not question or not question.strip():
        return "", "", ""
    t0 = time.time()
    base, ft = search(question, "base"), search(question, "finetuned")
    ms = (time.time() - t0) * 1000
    base_top = {base[0][0]["chunk_id"]} if base else set()
    ft_top = {ft[0][0]["chunk_id"]} if ft else set()
    verdict = ("두 모델의 1순위 결과가 동일합니다." if base_top == ft_top
               else "두 모델의 1순위 결과가 다릅니다 — 도메인 적응의 효과를 확인해 보세요.")
    note = f"<div style='color:#57606a;font-size:13px'>{verdict} (검색 {ms:.0f} ms)</div>"
    return render(base), render(ft), note


CSS = """
.gradio-container {max-width: 1150px !important}
#banner {background:#fff8c5;border:1px solid #d4a72c;border-radius:8px;padding:10px 14px;margin-bottom:6px}
"""

with gr.Blocks(title="근로 Q&A 검색 — 파인튜닝 전/후 비교") as demo:
    gr.HTML("<div id='banner'><b>⚠️ 포트폴리오용 연구 프로젝트이며 법률 자문이 아닙니다.</b><br>"
            "This is a portfolio research project, not legal advice. "
            "법령 스냅샷 기준일: 2026-09-25.</div>")
    gr.Markdown(
        "## 근로 Q&A 도메인 적응 — 검색 결과 비교\n"
        "일상적인 한국어 노동 질문을 입력하면, **기본 임베딩 모델**과 **노동법으로 파인튜닝한 모델**이 "
        "각각 어떤 법령·판례를 찾아오는지 나란히 보여줍니다. "
        "말뭉치: 법률 11종(시행령·시행규칙 포함) + 대법원 판례 507건 = 3,922개 청크.\n\n"
        "검증용 질문 335건 기준 **Recall@1 0.513 → 0.657 (상대 +28.1%)**.\n\n"
        "ℹ️ 유사도 점수는 **모델마다 척도가 다르므로 두 열 사이에서 직접 비교할 수 없습니다**. "
        "각 열 안에서의 순위만 의미가 있습니다."
    )
    q = gr.Textbox(label="질문", placeholder="예: 주 15시간 일하는 알바인데 주휴수당 받을 수 있나요?", lines=2)
    with gr.Row():
        btn = gr.Button("검색", variant="primary", scale=1)
        clear = gr.Button("지우기", scale=1)
    note = gr.HTML()
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 기본 모델 (multilingual-e5-base)")
            out_base = gr.HTML()
        with gr.Column():
            gr.Markdown("### 파인튜닝 모델 (this work)")
            out_ft = gr.HTML()
    gr.Examples(examples=[[e] for e in EXAMPLES], inputs=q, label="예시 질문 (클릭)")
    gr.Markdown(
        "---\n"
        "데이터 출처: 법제처 국가법령정보 OpenAPI (via [legalize-kr](https://github.com/legalize-kr/legalize-kr)). "
        "법령·판례 원문은 저작권법 제7조에 따라 보호받지 못하는 저작물입니다. "
        "코드: [GitHub](https://github.com/Habibulo/labor-law-rag-finetune)"
    )
    btn.click(compare, q, [out_base, out_ft, note])
    q.submit(compare, q, [out_base, out_ft, note])
    clear.click(lambda: ("", "", ""), None, [out_base, out_ft, note])

if __name__ == "__main__":
    # Gradio 6 moved theme/css from Blocks() to launch().
    demo.launch(css=CSS, theme=gr.themes.Soft())
