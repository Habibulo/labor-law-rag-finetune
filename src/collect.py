"""Phase 1: collect target laws and labor precedents from the legalize-kr GitHub mirrors.

Usage:
    python src/collect.py

1. Shallow, sparse, blob-less clones of legalize-kr and precedent-kr into cache_dir.
2. Laws: match target names to kr/ folders, keep one file per kind (법률/시행령/시행규칙),
   and pin each file to the version in force on snapshot_date using the file's commit history.
3. Precedents: select labor cases by court, case type, date, required sections, and 참조조문.
4. Copy the selection to data/raw/ and write data/raw/SOURCE.md and data/raw/manifest.json.

Set GITHUB_TOKEN to raise the GitHub API limit (unauthenticated: 60 requests/hour; this script
needs about one request per law file that has a future 시행일자).
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

from common import ROOT, load_config, split_frontmatter

sys.stdout.reconfigure(encoding="utf-8")

RAW = ROOT / "data" / "raw"


def git(args, cwd=None):
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)
    return out.stdout.decode("utf-8")


def ensure_clone(url, dest):
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {url} (shallow, sparse, blob-less) ...")
        git(["clone", "--depth", "1", "--filter=blob:none", "--sparse",
             "--config", "core.autocrlf=false", url + ".git", str(dest)])
    sha, commit_date = git(["log", "-1", "--format=%H%x09%cI"], cwd=dest).strip().split("\t")
    return sha, commit_date


def ls_tree(repo, path=None, recursive=False):
    args = ["ls-tree", "-z", "--name-only"] + (["-r"] if recursive else []) + ["HEAD"]
    if path:
        args.append(path)
    return [p for p in git(args, cwd=repo).split("\0") if p]


def http_get(url, api=False):
    headers = {"User-Agent": "labor-law-rag-finetune"}
    if api and os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    # GitHub answers 500 while it computes path history on a large repo for the first time;
    # it succeeds once warmed, so back off generously.
    waits = [5, 15, 30, 60, 120]
    for attempt in range(len(waits) + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                return r.read().decode("utf-8")
        except Exception as e:
            if attempt == len(waits):
                raise
            print(f"  retry in {waits[attempt]}s after error: {e}")
            time.sleep(waits[attempt])


def in_force_version(repo_url, path, snapshot, cache_dir):
    """Walk the file's commit history (newest first) and return the first version whose
    시행일자 <= snapshot, plus the newer versions that were skipped."""
    owner_repo = repo_url.removeprefix("https://github.com/")
    quoted = urllib.parse.quote(path)
    tag = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    skipped = []
    page = 1
    while True:
        cached_api = cache_dir / "history" / f"commits_p{page}_{tag}.json"
        if cached_api.exists():
            commits = json.loads(cached_api.read_text(encoding="utf-8"))
        else:
            # per_page must stay small: larger pages time out (HTTP 500) on files with long history.
            api = f"https://api.github.com/repos/{owner_repo}/commits?path={quoted}&per_page=3&page={page}"
            commits = json.loads(http_get(api, api=True))
            cached_api.parent.mkdir(parents=True, exist_ok=True)
            cached_api.write_text(json.dumps(commits), encoding="utf-8")
        if not commits:
            return None, None, skipped
        for c in commits:
            sha = c["sha"]
            cached = cache_dir / "history" / f"{sha}_{tag}.md"
            if cached.exists():
                text = cached.read_text(encoding="utf-8")
            else:
                text = http_get(f"https://raw.githubusercontent.com/{owner_repo}/{sha}/{quoted}")
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(text, encoding="utf-8")
            fm, _ = split_frontmatter(text)
            if str(fm["시행일자"]) <= snapshot:
                return sha, text, skipped
            skipped.append({"commit": sha, "공포일자": str(fm["공포일자"]), "시행일자": str(fm["시행일자"])})
        page += 1


def collect_laws(cfg, repo, head_sha):
    snapshot = cfg["snapshot_date"]
    folders = {unicodedata.normalize("NFC", p.split("/", 1)[1]): p for p in ls_tree(repo, "kr/")}
    wanted, missing = {}, []
    for name in cfg["laws"]:
        key = unicodedata.normalize("NFC", re.sub(r"\s+", "", name))
        if key in folders:
            wanted[name] = folders[key]
        else:
            missing.append(name)
    git(["sparse-checkout", "set", "--cone", *wanted.values()], cwd=repo)

    out_root = RAW / "laws"
    shutil.rmtree(out_root, ignore_errors=True)
    records, excluded = [], []
    for name, folder in wanted.items():
        by_kind = {}
        for f in sorted((repo / folder).glob("*.md")):
            kind = f.stem.split("(")[0]
            text = f.read_text(encoding="utf-8")
            fm, _ = split_frontmatter(text)
            entry = {"law": name, "kind": kind, "path": f"{folder}/{f.name}", "text": text,
                     "법령ID": str(fm["법령ID"]), "제목": fm["제목"],
                     "공포일자": str(fm["공포일자"]), "시행일자": str(fm["시행일자"])}
            if kind not in cfg["law_kinds"]:
                excluded.append({**entry, "reason": f"kind '{kind}' not in law_kinds"})
                continue
            by_kind.setdefault(kind, []).append(entry)
        for kind, entries in by_kind.items():
            entries.sort(key=lambda e: e["공포일자"], reverse=True)
            for e in entries[1:]:
                excluded.append({**e, "reason": f"older duplicate of {kind}; kept {entries[0]['path']}"})
            keep = entries[0]
            keep.update(commit=head_sha, pinned=False, skipped_future_versions=[])
            if keep["시행일자"] > snapshot:
                sha, text, skipped = in_force_version(cfg["sources"]["laws_repo"], keep["path"],
                                                      snapshot, repo.parent)
                keep["skipped_future_versions"] = skipped
                if sha:
                    fm, _ = split_frontmatter(text)
                    keep.update(commit=sha, pinned=True, text=text,
                                공포일자=str(fm["공포일자"]), 시행일자=str(fm["시행일자"]))
                else:
                    keep["pin_failed"] = True
            dest = out_root / folder.split("/", 1)[1] / f"{kind}.md"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(keep["text"], encoding="utf-8")
            keep["raw_path"] = dest.relative_to(ROOT).as_posix()
            records.append(keep)
    for r in records + excluded:
        r.pop("text", None)
    return records, excluded, missing


def sections(body):
    out, cur = {}, None
    for line in body.split("\n"):
        if line.startswith("## "):
            cur = line[3:].strip()
            out[cur] = []
        elif cur:
            out[cur].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


def collect_precedents(cfg, repo):
    pc = cfg["precedents"]
    cands = []
    for p in ls_tree(repo, recursive=True):
        parts = p.split("/")
        if len(parts) != 3:
            continue
        ctype, court, fname = parts
        if ctype in pc["case_types"] and court in pc["courts"] and fname.split("_")[1] >= pc["min_date"]:
            cands.append(p)
    groups = sorted({(p.split("/")[0], p.split("/")[1], p.split("_")[1][:4]) for p in cands})
    patterns = [f"/{t}/{c}/*_{y}-*" for t, c, y in groups]
    print(f"Precedent candidates by path: {len(cands)}; checking out {len(patterns)} year patterns ...")
    git(["sparse-checkout", "set", "--no-cone", *patterns], cwd=repo)

    pats = pc["ref_law_patterns"]
    counts = Counter()
    law_hits = Counter()
    selected = []
    for p in cands:
        fm, body = split_frontmatter((repo / p).read_text(encoding="utf-8"))
        sec = sections(body)
        if any(not sec.get(s) for s in pc["require_sections"]):
            counts["dropped_missing_section"] += 1
            continue
        refs = re.sub(r"\s+", "", sec["참조조문"]).replace("·", "ㆍ")
        hits = [pat for pat in pats if pat in refs]
        if not hits:
            counts["dropped_no_labor_ref"] += 1
            continue
        law_hits.update(hits)
        selected.append((str(fm.get("선고일자", "")), p))
    counts["matched"] = len(selected)
    selected.sort(reverse=True)
    if len(selected) > pc["max_cases"]:
        counts["dropped_over_cap"] = len(selected) - pc["max_cases"]
        selected = selected[: pc["max_cases"]]

    out_root = RAW / "precedents"
    shutil.rmtree(out_root, ignore_errors=True)
    for _, p in selected:
        dest = out_root / p.split("/")[0] / p.split("/")[2]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / p, dest)
    counts["candidates"] = len(cands)
    counts["selected"] = len(selected)
    return [p for _, p in selected], dict(counts), dict(law_hits)


def write_source_md(cfg, meta, laws, excluded, missing, prec_counts, law_hits):
    lines = [
        "# Raw data source record",
        "",
        "Generated by `src/collect.py`. The upstream repos may force-push (all hashes change),",
        "so the files in this directory are the source of truth for this project.",
        "",
        f"- Download date: {meta['download_date']}",
        f"- Law snapshot date (version in force): {cfg['snapshot_date']}",
        "",
        "| Repo | URL | HEAD commit | HEAD commit date |",
        "|---|---|---|---|",
        f"| legalize-kr | {cfg['sources']['laws_repo']} | `{meta['laws_sha']}` | {meta['laws_date']} |",
        f"| precedent-kr | {cfg['sources']['precedents_repo']} | `{meta['prec_sha']}` | {meta['prec_date']} |",
        "",
        "Both repos are generated from the 국가법령정보센터 OpenAPI (https://open.law.go.kr).",
        "Their READMEs state: 원문 = 공공저작물, repo structure/metadata = MIT.",
        "",
        "## Laws",
        "",
        "| Law | Kind | Source path | Commit | 공포일자 | 시행일자 | Pinned to in-force version |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in laws:
        pinned = "yes" if r["pinned"] else ("FAILED" if r.get("pin_failed") else "no (HEAD in force)")
        lines.append(f"| {r['law']} | {r['kind']} | `{r['path']}` | `{r['commit'][:10]}` | "
                     f"{r['공포일자']} | {r['시행일자']} | {pinned} |")
    lines += ["", "Not-yet-effective versions skipped (promulgated but 시행일자 after the snapshot date):", ""]
    for r in laws:
        for s in r["skipped_future_versions"]:
            lines.append(f"- {r['law']} {r['kind']}: 공포 {s['공포일자']}, 시행 {s['시행일자']} (`{s['commit'][:10]}`)")
    lines += ["", "Excluded law files:", ""]
    lines += [f"- `{e['path']}` ({e['제목']}, 법령ID {e['법령ID']}, 공포 {e['공포일자']}): {e['reason']}"
              for e in excluded] or ["- none"]
    lines += ["", "Laws not found: " + (", ".join(missing) if missing else "none"), ""]
    pc = cfg["precedents"]
    lines += [
        "## Precedents",
        "",
        f"Selection: courts={pc['courts']}, case_types={pc['case_types']}, 선고일자 >= {pc['min_date']}, "
        f"non-empty {pc['require_sections']}, 참조조문 mentions a target law, newest first, cap {pc['max_cases']}.",
        "",
        "| Step | Count |",
        "|---|---|",
    ]
    for k in ["candidates", "dropped_missing_section", "dropped_no_labor_ref", "matched",
              "dropped_over_cap", "selected"]:
        lines.append(f"| {k} | {prec_counts.get(k, 0)} |")
    lines += ["", "Selected cases per matched law pattern (a case can match several):", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(law_hits.items(), key=lambda kv: -kv[1])]
    (RAW / "SOURCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    cfg = load_config()
    cache = ROOT / cfg["sources"]["cache_dir"]
    laws_repo, prec_repo = cache / "legalize-kr", cache / "precedent-kr"
    laws_sha, laws_date = ensure_clone(cfg["sources"]["laws_repo"], laws_repo)
    prec_sha, prec_date = ensure_clone(cfg["sources"]["precedents_repo"], prec_repo)

    laws, excluded, missing = collect_laws(cfg, laws_repo, laws_sha)
    print(f"Laws: {len(laws)} files kept, {len(excluded)} excluded, not found: {missing or 'none'}")
    for r in laws:
        flag = " (pinned to in-force version)" if r["pinned"] else ""
        print(f"  {r['law']} / {r['kind']}: 시행일자 {r['시행일자']}{flag}")

    prec_paths, prec_counts, law_hits = collect_precedents(cfg, prec_repo)
    print(f"Precedents: {prec_counts}")

    meta = {"download_date": date.today().isoformat(), "laws_sha": laws_sha, "laws_date": laws_date,
            "prec_sha": prec_sha, "prec_date": prec_date}
    manifest = {**meta, "snapshot_date": cfg["snapshot_date"], "laws": laws, "excluded_law_files": excluded,
                "laws_not_found": missing, "precedent_counts": prec_counts, "precedent_law_hits": law_hits,
                "precedents": prec_paths}
    (RAW / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_source_md(cfg, meta, laws, excluded, missing, prec_counts, law_hits)
    print(f"Wrote {RAW / 'SOURCE.md'} and {RAW / 'manifest.json'}")


if __name__ == "__main__":
    main()
