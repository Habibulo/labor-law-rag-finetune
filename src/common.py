from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path="configs/data.yaml"):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def load_env(path=".env"):
    """Read KEY=VALUE lines from the project .env (gitignored) without printing them."""
    f = ROOT / path
    if not f.exists():
        return {}
    pairs = (l.split("=", 1) for l in f.read_text(encoding="utf-8-sig").splitlines() if "=" in l)
    return {k.strip(): v.strip() for k, v in pairs}


def read_jsonl(path):
    import json
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def split_frontmatter(text):
    """Return (frontmatter dict, body) for Markdown with a leading YAML block."""
    text = text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        return {}, text
    end = text.index("\n---\n", 4)
    return yaml.safe_load(text[4:end]), text[end + 5:]
