"""Free-tier LLM calls with a model fallback chain (Gemini API + Groq), JSON-schema output.

Lessons built in (see docs/decisions_log.md #28, #31, #32):
- a new seed on every retry: with a fixed seed a model reproduces the same failure;
- a tight output cap and finish-reason check: flash-lite sometimes loops until MAX_TOKENS;
- per-model daily quota tracking: exhausted models are skipped, stop only when all are;
- per-model cooldowns: a model that is overloaded (5xx) or hits a per-minute limit (429) rests
  while later requests stay on the next working model (decisions_log #38).
"""
import json
import re
import time
import urllib.error
import urllib.request

GEMINI_API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# OpenAI-compatible providers, addressed in configs as "<prefix>:<model name>".
# Adding a provider is one entry here plus its key in .env — no code change.
PROVIDERS = {
    "groq": {"url": "https://api.groq.com/openai/v1/chat/completions", "key": "GROQ_API_KEY",
             "extra": {"reasoning_effort": "low"}},
    "nvidia": {"url": "https://integrate.api.nvidia.com/v1/chat/completions", "key": "NVIDIA_API_KEY",
               "extra": {}},
}


class DailyQuotaExceeded(Exception):
    pass


def api_keys(env):
    """Every provider key present in .env, plus the Gemini key. Missing ones are simply absent,
    and LLMChain then skips the models that need them."""
    names = ["GEMINI_API_KEY"] + [p["key"] for p in PROVIDERS.values()]
    return {n: env[n] for n in names if env.get(n)}


def to_json_schema(s):
    """Gemini-style schema (uppercase types) -> standard JSON Schema for OpenAI-compatible APIs."""
    out = {k: v for k, v in s.items() if k not in ("type", "properties", "items")}
    out["type"] = s["type"].lower()
    if "properties" in s:
        out["properties"] = {k: to_json_schema(v) for k, v in s["properties"].items()}
        out["additionalProperties"] = False
    if "items" in s:
        out["items"] = to_json_schema(s["items"])
    return out


def _post(url, body, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "labor-law-rag-finetune", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _parse_error(e):
    try:
        err = json.loads(e.read()).get("error", {})
    except Exception:
        return {}, ""
    return err, json.dumps(err, ensure_ascii=False)


RETRY_IN_RE = re.compile(r"try again in\s+(?:(\d+)m)?([\d.]+)s", re.I)


def _retry_delay(err, headers, default):
    """Gemini puts the wait in error.details[RetryInfo]; Groq only in the message text
    ("Please try again in 21m5.76s"); some providers use a retry-after header."""
    for d in err.get("details", []):
        if d.get("@type", "").endswith("RetryInfo") and d.get("retryDelay", "").endswith("s"):
            return float(d["retryDelay"][:-1]) + 1
    m = RETRY_IN_RE.search(err.get("message", ""))
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2)) + 1
    if headers and headers.get("retry-after"):
        try:
            return float(headers["retry-after"]) + 1
        except ValueError:
            pass
    return default


class LLMChain:
    """cfg keys: models, temperature, max_output_tokens, groq_max_output_tokens, max_retries,
    min_seconds_between_requests. Models prefixed 'groq:' use Groq; others the Gemini API."""

    def __init__(self, cfg, keys, state_path=None):
        self.cfg, self.keys = cfg, keys
        self.models = [m for m in cfg["models"] if self._usable(m, keys)]
        missing = [m for m in cfg["models"] if m not in self.models]
        if missing:
            print(f"  skipping (no API key in .env): {', '.join(missing)}")
        self.exhausted = set()          # daily quota used up (or model unusable): skipped
        self.cooldown_until = {}        # model -> time it may be tried again
        self.fail_streak = {}           # consecutive overload failures, for growing cooldowns
        self._last_call = 0.0
        # A slow/queued model must not block the run: NVIDIA NIM can hang well past a minute.
        self._timeout = cfg.get("request_timeout_s", 120)
        # Daily-quota benches are saved to disk so a restarted script does not retry those models.
        self.state_path = state_path
        self.daily_rest_s = cfg.get("daily_quota_rest_hours", 6) * 3600
        self._benched = {}
        if state_path and state_path.exists():
            self._benched = {m: t for m, t in json.loads(state_path.read_text(encoding="utf-8")).items()
                             if t > time.time()}
            for m, t in self._benched.items():
                if m in self.models:
                    self.exhausted.add(m)
                    print(f"  {m}: daily quota benched until {time.strftime('%Y-%m-%d %H:%M', time.localtime(t))}")

    @staticmethod
    def _usable(model, keys):
        """A model is usable only if the key its provider needs is present."""
        p = PROVIDERS.get(model.split(":", 1)[0])
        return bool(keys.get(p["key"])) if p else bool(keys.get("GEMINI_API_KEY"))

    def _bench_for_day(self, model, retry_after=None):
        """Bench a model whose DAILY quota is gone.

        Some daily quotas are rolling windows that free up sooner than a calendar day: Groq's TPD
        error says "try again in 21m" and means it. Gemini's daily error also carries a RetryInfo,
        but a misleading one (18s for a quota that resets at midnight PT), so a delay under
        `min_trusted_retry_s` is ignored in favour of the configured rest.
        """
        rest = self.daily_rest_s
        if retry_after and retry_after >= self.cfg.get("min_trusted_retry_s", 60):
            rest = min(retry_after, self.daily_rest_s)
        self.exhausted.add(model)
        self._benched[model] = time.time() + rest
        until = time.strftime('%H:%M', time.localtime(self._benched[model]))
        print(f"  {model}: daily quota used up; benched until {until} ({rest / 60:.0f} min)")
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self._benched, indent=1), encoding="utf-8")

    def _throttle(self):
        time.sleep(max(0.0, self.cfg["min_seconds_between_requests"] - (time.time() - self._last_call)))
        self._last_call = time.time()

    def _call(self, model, system, prompt, schema, seed):
        """Return (json_text or None if the answer did not finish normally, finish_reason, version)."""
        c = self.cfg
        prefix = model.split(":", 1)[0]
        if prefix in PROVIDERS:
            p = PROVIDERS[prefix]
            name = model.split(":", 1)[1]
            out = _post(p["url"], {
                "model": name, "temperature": c["temperature"], "seed": seed,
                "max_completion_tokens": c["groq_max_output_tokens"],
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "output", "schema": to_json_schema(schema)}},
                **p["extra"],
            }, {"Authorization": f"Bearer {self.keys[p['key']]}"}, self._timeout)
            ch = out["choices"][0]
            ok = ch["finish_reason"] == "stop"
            return (ch["message"]["content"] if ok else None), ch["finish_reason"], out.get("model", name)
        out = _post(GEMINI_API.format(model=model), {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": c["temperature"], "seed": seed,
                                 "maxOutputTokens": c["max_output_tokens"],
                                 "responseMimeType": "application/json", "responseSchema": schema},
        }, {"x-goog-api-key": self.keys["GEMINI_API_KEY"]}, self._timeout)
        cand = out["candidates"][0]
        ok = cand.get("finishReason") == "STOP"
        return (cand["content"]["parts"][-1]["text"] if ok else None), cand.get("finishReason"), \
            out.get("modelVersion", model)

    def _rest(self, model, seconds, why):
        """Bench a model: it is skipped until the cooldown ends, so later requests stay on the
        next working model instead of retrying a model that just failed."""
        self.cooldown_until[model] = time.time() + seconds
        print(f"  {model}: {why}; resting {seconds:.0f}s, using the next model meanwhile")

    def _available(self):
        now = time.time()
        return [m for m in self.models if m not in self.exhausted and self.cooldown_until.get(m, 0) <= now]

    def generate_json(self, system, prompt, schema, base_seed, validate=None):
        """Return (parsed_json, model, model_version, seed) or None if every attempt failed.
        `validate(parsed)` may raise KeyError/ValueError to reject a malformed answer.
        Raises DailyQuotaExceeded when every model's daily quota is used up.

        Models are tried in config order, skipping exhausted (daily quota; whole run) and resting
        (per-minute limit or overloaded; cooldown) ones. An attempt = one pass over the models that
        are available; a bad answer moves on without a cooldown (a new seed usually fixes it)."""
        attempt = 0
        while attempt < self.cfg["max_retries"]:
            if all(m in self.exhausted for m in self.models):
                back = min(self._benched.values()) if self._benched else None
                when = time.strftime('%Y-%m-%d %H:%M', time.localtime(back)) if back else "tomorrow"
                raise DailyQuotaExceeded(f"all models benched; first one back at {when}")
            available = self._available()
            if not available:
                wait = min(self.cooldown_until[m] for m in self.models if m not in self.exhausted) - time.time()
                print(f"  all models resting; waiting {max(wait, 0):.0f}s")
                time.sleep(max(wait, 0) + 1)
                continue
            seed = base_seed + attempt * 7919
            for model in available:
                self._throttle()
                try:
                    text, finish, version = self._call(model, system, prompt, schema, seed)
                    if text is None:
                        print(f"  {model} finished with {finish}; trying next model")
                        continue
                    parsed = json.loads(text)
                    if validate:
                        validate(parsed)
                    self.fail_streak[model] = 0
                    return parsed, model, version, seed
                except urllib.error.HTTPError as e:
                    headers = e.headers
                    err, raw = _parse_error(e)
                    if e.code == 429:
                        # Gemini's daily-quota 429 also says "retry in 18s"; the quotaId is what counts.
                        if "PerDay" in raw or "per day" in raw.lower():
                            self._bench_for_day(model, _retry_delay(err, headers, None))
                        else:
                            self._rest(model, _retry_delay(err, headers, 600), "per-minute limit (429)")
                        continue
                    if e.code in (500, 502, 503, 504):
                        self.fail_streak[model] = self.fail_streak.get(model, 0) + 1
                        seconds = min(600 * 2 ** (self.fail_streak[model] - 1), 900)
                        self._rest(model, seconds, f"overloaded ({e.code} {err.get('status', '')})")
                        continue
                    if e.code in (400, 404):
                        self.exhausted.add(model)
                        print(f"  {model}: HTTP {e.code} ({raw[:150]}); disabled for this run")
                        continue
                    raise RuntimeError(f"HTTP {e.code} on {model}: {raw[:300]}")
                except (KeyError, IndexError, ValueError, TypeError) as e:
                    print(f"  bad response from {model}: {type(e).__name__}; trying next model")
                    continue
                except (urllib.error.URLError, TimeoutError) as e:
                    self.fail_streak[model] = self.fail_streak.get(model, 0) + 1
                    self._rest(model, min(600 * 2 ** (self.fail_streak[model] - 1), 900), f"network error ({e})")
                    continue
            attempt += 1
        return None
