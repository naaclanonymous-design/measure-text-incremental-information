"""ZERO-SHOT operator inference with hosted LLMs (Gemini-3.1 / GPT-5.4, thinking OFF) on the
influenza TEST set (or --split val: the VALIDATION set), using the finetune pipeline's own prompts.

Two INPUT variants (--text):
  keep (default) : 12-week history + 3-week Chronos-2 baseline + recent baseline errors + the
                   PLAIN CDC FluView weekly reports of the last --text-weeks (default 5) weeks
                   (origin week and the 4 before it), exactly as the corpus's weekly_text_block
                   renders them (date-scrubbed, relative labels) — NO generated CoT anywhere.
                   Reports missing from the corpus stay missing (the block lists only what exists).
                   = the input of the standard finetune.py runs (scale/, original/).
  drop           : the same numbers, with the report section replaced by the notext stub
                   = the input of the scale_residualcot_notext finetune arm.

NEW script — reads the corpus and the finetune prompt machinery from finetune.py, touches no
existing run dir, and writes only under its own --output-root.

Input reproduction (byte-identical to the finetune runs):
  * same corpus rows / split composition / skip rules  (finetune.load_examples)
  * same finetune prompt template + influenza wording   (finetune.ensure_finetune_prompt,
    finetune.PROMPT_WORDING, finetune.build_user_prompt)
  * --text drop: weekly_text_block replaced by the notext stub (finetune_residual_cot.NOTEXT_STUB)
  * --text keep: weekly_text_block kept, trimmed to the last --text-weeks weeks (the corpus
    already holds at most 5, so with the default this is a no-op that just enforces the spec)
  The rebuilt test prompts are checked against an existing finetune run's test.jsonl
  (--reference-test-jsonl; default = the scale_residualcot_notext run for drop, the Qwen3-8B
  scale run for keep) and the script refuses to proceed on any mismatch unless --no-verify.
  System prompt: finetune.SYSTEM_PROMPT with the Qwen-only "/no_think" control token removed
  (--keep-no-think retains it verbatim).

Models / thinking:
  gemini : gemini-3.1-pro-preview via Vertex AI (ADC, GOOGLE_CLOUD_PROJECT). Gemini-3.x has NO
           thinking off-switch (the API rejects thinking_level=minimal for 3.1-pro); the floor
           thinking_level=low is used and the per-call thoughts_token_count is recorded.
  gpt    : gpt-5.4 via the OpenAI chat API with reasoning_effort="none" (0 reasoning tokens;
           recorded per call from usage.completion_tokens_details.reasoning_tokens).
  Sampling mirrors the pipeline's swift infer: temperature 0 (GPT falls back to the API
  default if the model rejects temperature), max output tokens = finetune.MAX_NEW_TOKENS.

Evaluation: finetune.evaluate — identical categorical metrics (direction / bin / operator
accuracy) and the rough count-based MAE/MSE vs the Chronos-2 baseline, with the chronos2_full
±10% relative no-change band. Overall AND per-state (GA / NJ / NY) metrics are written.

Output (default --output-root <paper-content>/output/zero_shot):
  <output-root>/influenza_TSFM/chronos2_full/GA_NY_NJ-gemini/<variant>[/val]/
      variant = text5wk_input (--text keep, 5 weeks) | scale_residualcot_notext_input (--text drop)
      test-split results live at the variant root; --split val writes to the val/ subdir
      test.jsonl                       the exact prompts sent (system + user), one per test row
      <model_tag>/responses.jsonl      raw responses + token usage (resumable; one row per id)
      <model_tag>/predictions.csv      parsed operator vs truth, per row (finetune.evaluate)
      <model_tag>/metrics.json         overall + per-state metrics, config, thinking settings
      summary.json / summary.csv       side-by-side table across the models run

Run (forecast conda env has google-genai + openai):
  python zero_shot_api_notext.py --providers gemini,gpt   # text keep, 5 wk
  ... --text drop    the no-text variant (scale_residualcot_notext input)
  ... --split val     the validation split (verified against the finetune runs' val_infer.jsonl)
  ... --limit 2      smoke test (first 2 test rows per model)
  ... --providers gpt --gpt-model gpt-5.4 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_CODE_ROOT = Path(__file__).resolve().parents[1]          # <paper-content>/code
for _d in ("cot_generation", "finetune", "zero_shot"):
    sys.path.insert(0, str(_CODE_ROOT / _d))
sys.path.insert(0, str(_CODE_ROOT))                         # paper_paths.py
import paper_paths as P                                     # noqa: E402
import finetune as ft                                   # noqa: E402
from finetune_residual_cot import NOTEXT_STUB           # noqa: E402

OUTPUT_ROOT = P.ZERO_SHOT_OUTPUT_ROOT
# finetune run whose test.jsonl carries the same input, per --text (the paper's best influenza
# run). Used only as a byte-identity check of the rebuilt prompts; pass --no-verify (or
# --reference-test-jsonl) when that run does not exist locally.
FT_ROOT = P.FINETUNE_OUTPUT_ROOT / "influenza" / "chronos2_full" / "GA_NY_NJ-gemini"
REFERENCE_RUN = {
    "drop": None,
    "keep": FT_ROOT / "scale/Qwen3-8B_scale_lr0.0001_sched4-1to0.5-4/test.jsonl",
}
TEXT_WEEKS = 5                       # keep the reports of the origin week + the 4 before it
GEMINI_MODEL = "gemini-3.1-pro-preview"
GEMINI_THINKING_LEVEL = "low"        # gemini-3.x floor; 'minimal' is rejected by 3.1-pro
GPT_MODEL = "gpt-5.4"
GPT_REASONING_EFFORT = "none"        # thinking off
OPENAI_CRED_PATH = P.OPENAI_CRED_PATH   # {"api_key": "..."}; or export OPENAI_API_KEY
TEMPERATURE = 0.0                    # swift infer used --temperature 0 --top_p 1.0
INPUT_SUBDIR = {"drop": "scale_residualcot_notext_input", "keep": "text{n}wk_input"}
_REPORT_HDR = re.compile(r"^--- (?P<kind>.+?) for (?P<rel>.+?) ---$", re.M)


# ══════════════════════════════════════════════════════════════════════════════
# Provider clients — one .generate(system, user) -> dict(text, usage, finish) interface
# ══════════════════════════════════════════════════════════════════════════════
class GeminiZS:
    def __init__(self, model: str, thinking_level: str, max_tokens: int, temperature: float,
                 thinking_budget: Optional[int] = None):
        from google import genai
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            sys.exit("export GOOGLE_CLOUD_PROJECT (Vertex AI ADC; see cot_common.GeminiClient)")
        self.model, self.level = model, thinking_level
        # gemini-2.5-* reject thinking_level and take an integer thinking_budget instead
        # (2.5-pro floor 128, cannot be 0; 2.5-flash accepts 0 = off). When a budget is given
        # it REPLACES thinking_level in the request and in the run tag.
        self.budget = thinking_budget
        self.max_tokens, self.temperature = max_tokens, temperature
        self._client = genai.Client(vertexai=True, project=project, location="global")

    def tag(self) -> str:
        if self.budget is not None:
            return f"{self.model}_budget-{self.budget}"
        return f"{self.model}_think-{self.level}"

    def config(self) -> Dict:
        if self.budget is not None:
            think = {"thinking_budget": self.budget,
                     "thinking_note": ("gemini-2.5-* take thinking_budget (tokens); 128 is the "
                                       "2.5-pro floor (budget 0 rejected), used as 'low'")}
        else:
            think = {"thinking_level": self.level,
                     "thinking_note": ("gemini-3.x cannot disable thinking; 'low' is the API floor "
                                       "(thinking_level=minimal is rejected for this model)")}
        return {"provider": "gemini", "model": self.model, **think,
                "temperature": self.temperature, "top_p": 1.0,
                "max_output_tokens": self.max_tokens, "endpoint": "vertexai/global"}

    def generate(self, system: str, user: str, retries: int = 4, backoff_s: float = 5.0) -> Dict:
        from google.genai import types as T
        tc = (T.ThinkingConfig(thinking_budget=self.budget) if self.budget is not None
              else T.ThinkingConfig(thinking_level=self.level))
        cfg = T.GenerateContentConfig(
            temperature=self.temperature, top_p=1.0, max_output_tokens=self.max_tokens,
            system_instruction=system, thinking_config=tc)
        last = None
        for a in range(retries):
            try:
                r = self._client.models.generate_content(model=self.model, contents=user, config=cfg)
                txt = (r.text or "").strip()
                um = r.usage_metadata
                fin = None
                if r.candidates:
                    fr = r.candidates[0].finish_reason
                    fin = getattr(fr, "value", str(fr))
                if not txt:
                    raise RuntimeError(f"empty response text (finish_reason={fin})")
                return {"text": txt, "finish_reason": fin,
                        "usage": {"prompt_tokens": um.prompt_token_count,
                                  "thinking_tokens": um.thoughts_token_count or 0,
                                  "output_tokens": um.candidates_token_count,
                                  "total_tokens": um.total_token_count}}
            except Exception as e:                      # noqa: BLE001
                last = e
                wait = backoff_s * (2 ** a)
                print(f"    [gemini retry {a + 1}/{retries}] {str(e)[:200]} — sleep {wait:.0f}s",
                      flush=True)
                time.sleep(wait)
        raise RuntimeError(f"gemini {self.model} failed after {retries} attempts: {last}")


class GPTZS:
    def __init__(self, model: str, reasoning_effort: str, max_tokens: int, temperature: float,
                 cred_path: Path):
        from openai import OpenAI
        obj = json.loads(Path(cred_path).read_text())
        key = next((str(v) for k, v in obj.items() if v and "key" in k.lower()), None)
        if not key:
            sys.exit(f"no API key field in {cred_path}")
        self.model, self.effort = model, reasoning_effort
        self.max_tokens, self.temperature = max_tokens, temperature
        self._temp_ok = True          # dropped if the model rejects temperature
        self._lock = threading.Lock()
        self._client = OpenAI(api_key=key)

    def tag(self) -> str:
        return f"{self.model}_reasoning-{self.effort}"

    def config(self) -> Dict:
        return {"provider": "openai", "model": self.model, "reasoning_effort": self.effort,
                "thinking_note": "reasoning_effort=none disables thinking (0 reasoning tokens)",
                "temperature": self.temperature if self._temp_ok else "api-default (rejected)",
                "max_completion_tokens": self.max_tokens, "endpoint": "chat.completions"}

    def generate(self, system: str, user: str, retries: int = 4, backoff_s: float = 5.0) -> Dict:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last = None
        for a in range(retries):
            kw = {"model": self.model, "messages": msgs, "reasoning_effort": self.effort,
                  "max_completion_tokens": self.max_tokens}
            if self._temp_ok:
                kw["temperature"] = self.temperature
            try:
                r = self._client.chat.completions.create(**kw)
                ch = r.choices[0]
                txt = (ch.message.content or "").strip()
                u = r.usage
                det = getattr(u, "completion_tokens_details", None)
                if not txt:
                    raise RuntimeError(f"empty response text (finish_reason={ch.finish_reason})")
                return {"text": txt, "finish_reason": ch.finish_reason,
                        "usage": {"prompt_tokens": u.prompt_tokens,
                                  "thinking_tokens": (getattr(det, "reasoning_tokens", 0) or 0),
                                  "output_tokens": u.completion_tokens,
                                  "total_tokens": u.total_tokens}}
            except Exception as e:                      # noqa: BLE001
                last = e
                msg = str(e)
                if "insufficient_quota" in msg:
                    raise RuntimeError(f"OpenAI account has no quota: {e}") from e
                if "temperature" in msg and "unsupported" in msg.lower() and self._temp_ok:
                    with self._lock:
                        self._temp_ok = False
                    print("    [gpt] model rejects temperature — retrying without it", flush=True)
                    continue
                wait = backoff_s * (2 ** a)
                print(f"    [gpt retry {a + 1}/{retries}] {msg[:200]} — sleep {wait:.0f}s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"gpt {self.model} failed after {retries} attempts: {last}")


# ══════════════════════════════════════════════════════════════════════════════
# Data: the notext test prompts, rebuilt through finetune.py
# ══════════════════════════════════════════════════════════════════════════════
# covid_TSFM_bitong_org = the final-fold TEST block of the COVID dataset (TX/VT/NJ, 78 origins)
COVID_BITONG_MODES = ("covid_TSFM_bitong_org",)
DEFAULT_CORPUS = {"influenza": "gemini", "covid_TSFM_bitong_org": "gemini"}   # the released original corpora
# finetune run whose test.jsonl carries the same input, per (data_mode, --text)
_FT = P.FINETUNE_OUTPUT_ROOT
REFERENCE_RUN_BY_MODE = {
    "influenza": REFERENCE_RUN,
    "covid_TSFM_bitong_org": {"keep": None, "drop": None},   # no reference run shipped: check skipped
}


def resolve_corpus(input_root: Path, data_mode: str, mode: str, combo: str):
    run_dir = input_root / data_mode / mode / combo
    data_dir = data_mode
    if not (run_dir / "accepted.jsonl").exists() and data_mode == "influenza":
        alt = input_root / "influenza_TSFM" / mode / combo
        if (alt / "accepted.jsonl").exists():
            data_dir, run_dir = "influenza_TSFM", alt
    if not (run_dir / "accepted.jsonl").exists() and data_mode in ft.BITONG_MODES:
        # the TX/VT/NJ corpus was generated under covid_TSFM_updated (only bins.json is used)
        alt = input_root / ft.BITONG_CORPUS_DATA_MODE / mode / combo
        if (alt / "accepted.jsonl").exists():
            run_dir = alt
    if not (run_dir / "accepted.jsonl").exists():      # released corpus under data/cot_corpora
        found = P.find_corpus(input_root, data_mode, mode, combo)
        if found is not None:
            data_dir = found[0] if data_mode == "influenza" else data_dir
            run_dir = found[1]
    if not (run_dir / "accepted.jsonl").exists():
        sys.exit(f"corpus not found: {run_dir}/accepted.jsonl")
    return data_dir, run_dir


def load_split_examples(args, states: List[str]):
    """(data_dir, combo, run_dir, horizon, bins, examples) for args.split.
    influenza: the corpus rows of that split (finetune.load_examples).
    covid_TSFM_bitong_org: the fold CSVs — test = the FINAL test block (78 origins, 2025-10 ..
    2026-03), val = the fold-B + fold-C validation blocks — built offline exactly as
    finetune.build_bitong_org_splits does (same blocks, denorm params and count truths)."""
    combo = f"{ft.run_dir_name(states)}-{args.corpus_model.lower()}"
    data_dir, run_dir = resolve_corpus(args.input_root, args.data_mode, args.mode, combo)
    horizon = ft.horizon_for(args.data_mode, args.mode)
    bins = ft.load_bins(run_dir)
    if args.data_mode in COVID_BITONG_MODES:
        fold_root = ft.BITONG_FOLD_ROOT
        orig = ft.load_covid_orig(states, args.data_mode, ft.bitong_fold_dir(fold_root, "final"))
        if args.split == "test":
            exs = ft.build_covid_eval_examples(states, ["test"], horizon, orig, args.data_mode,
                                               ft.bitong_fold_dir(fold_root, ft.BITONG_ORG_TEST_FOLD)
                                               ).get("test", [])
        else:
            exs = []
            for f in ft.BITONG_ORG_VAL_FOLDS:
                exs += ft.build_covid_eval_examples(states, ["val"], horizon, orig, args.data_mode,
                                                    ft.bitong_fold_dir(fold_root, f)).get("val", [])
            exs.sort(key=lambda e: (e["state"], e["reference_date"]))
        data_dir = args.data_mode
    else:
        by_split = ft.load_examples(run_dir, horizon, bins, keep_states=set(states),
                                    data_mode=args.data_mode)
        exs = by_split.get(args.split, [])
    if not exs:
        sys.exit(f"no {args.split} examples for {args.data_mode}")
    return data_dir, combo, run_dir, horizon, bins, exs


def report_weeks_ago(rel: str) -> int:
    """'the most recent surveillance week (the forecast-origin week)' -> 0,
    '1 week before the forecast origin' -> 1, 'N weeks before ...' -> N."""
    if "most recent" in rel or "origin week" in rel:
        return 0
    m = re.match(r"(\d+) weeks? before", rel)
    return int(m.group(1)) if m else 99


def trim_text_block(block: str, n_weeks: int) -> str:
    """Keep only the reports from the last n_weeks weeks (weeks_ago < n_weeks), preserving
    the corpus's own rendering of each report (header + scrubbed text) and their order."""
    hdrs = list(_REPORT_HDR.finditer(block))
    if not hdrs:
        return block                                    # stub / no-report placeholder
    keep = []
    for i, h in enumerate(hdrs):
        end = hdrs[i + 1].start() if i + 1 < len(hdrs) else len(block)
        if report_weeks_ago(h.group("rel")) < n_weeks:
            keep.append(block[h.start():end].rstrip("\n") + "\n")
    return "\n".join(keep) if keep else block


def build_test_prompts(args, states: List[str]):
    data_dir, combo, run_dir, horizon, bins, test = load_split_examples(args, states)
    n_rep = []
    for ex in test:
        if args.text == "drop":                       # the notext ablation input
            ex["weekly_text_block"] = NOTEXT_STUB
        else:                                         # plain reports, last --text-weeks weeks
            ex["weekly_text_block"] = trim_text_block(ex["weekly_text_block"], args.text_weeks)
            n_rep.append(len(_REPORT_HDR.findall(ex["weekly_text_block"])))
    if n_rep:
        from collections import Counter
        print(f"  [text keep] reports per test row (last {args.text_weeks} weeks): "
              f"{sorted(Counter(n_rep).items())}")
    names_str = " | ".join(ft.bin_names(bins))
    template = ft.ensure_finetune_prompt(args.mode, args.prompt_dir, args.data_mode)
    wording = ft.PROMPT_WORDING[(args.data_mode, args.mode)]
    sysp = ft.system_prompt_for(args.data_mode)
    if not args.keep_no_think:
        sysp = sysp.replace(ft.NO_THINK, "").strip()
    prompts = [ft.build_user_prompt(ex, horizon, names_str, template, wording) for ex in test]
    return data_dir, combo, run_dir, horizon, bins, test, prompts, sysp


def verify_against_reference(test: List[Dict], prompts: List[str], ref: Path) -> None:
    if not ref.exists():
        print(f"  [verify] reference test.jsonl not found ({ref}); skipping check")
        return
    ref_rows = [json.loads(l) for l in ref.read_text().splitlines() if l.strip()]
    ref_users = [next(m["content"] for m in r["messages"] if m["role"] == "user") for r in ref_rows]
    if len(ref_users) != len(prompts):
        sys.exit(f"[verify] row count differs: rebuilt {len(prompts)} vs reference {len(ref_users)} "
                 f"({ref}); --no-verify to override")
    bad = [i for i, (a, b) in enumerate(zip(prompts, ref_users)) if a != b]
    if bad:
        i = bad[0]
        sys.exit(f"[verify] {len(bad)} prompts differ from the reference run (first: row {i}, "
                 f"id={test[i]['id']}); --no-verify to override")
    print(f"  [verify] all {len(prompts)} rebuilt user prompts are byte-identical to {ref}")


# ══════════════════════════════════════════════════════════════════════════════
# Inference (resumable) + evaluation
# ══════════════════════════════════════════════════════════════════════════════
def load_responses(path: Path) -> Dict[str, Dict]:
    out = {}
    if path.exists():
        for l in path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                out[r["id"]] = r
    return out


def run_model(client, tag: str, test: List[Dict], prompts: List[str], sysp: str,
              out_dir: Path, workers: int, force: bool) -> Dict[str, Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    resp_path = out_dir / "responses.jsonl"
    if force and resp_path.exists():
        resp_path.unlink()
    done = load_responses(resp_path)
    todo = [(ex, p) for ex, p in zip(test, prompts) if ex["id"] not in done]
    print(f"  [{tag}] {len(done)} cached, {len(todo)} to run, workers={workers}", flush=True)
    lock = threading.Lock()
    n_ok = n_fail = 0

    def one(ex, p):
        t0 = time.time()
        r = client.generate(sysp, p)
        return {"id": ex["id"], "state": ex["state"], "reference_date": ex["reference_date"],
                "model": client.model, "response": r["text"], "finish_reason": r["finish_reason"],
                "usage": r["usage"], "latency_s": round(time.time() - t0, 2)}

    with ThreadPoolExecutor(max_workers=workers) as pool, resp_path.open("a") as f:
        futs = {pool.submit(one, ex, p): ex for ex, p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            ex = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:                  # noqa: BLE001
                n_fail += 1
                print(f"    FAIL {ex['id']}: {str(e)[:200]}", flush=True)
                continue
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
                done[rec["id"]] = rec
            n_ok += 1
            if i % 10 == 0 or i == len(todo):
                u = rec["usage"]
                print(f"    {i}/{len(todo)} done  (last: {ex['state']} {ex['reference_date']} "
                      f"think={u['thinking_tokens']} out={u['output_tokens']} "
                      f"{rec['latency_s']}s)", flush=True)
    print(f"  [{tag}] finished: {n_ok} ok, {n_fail} failed, {len(done)}/{len(test)} total",
          flush=True)
    return done


def evaluate_model(tag: str, client_cfg: Dict, test: List[Dict], responses: Dict[str, Dict],
                   horizon: int, bins, band: float, out_dir: Path, states: List[str],
                   input_desc: str = "", split: str = "test") -> Dict:
    resp_list = [responses.get(ex["id"], {}).get("response", "") for ex in test]
    df, m = ft.evaluate(test, resp_list, horizon, bins, band)
    df["response"] = resp_list
    df.to_csv(out_dir / "predictions.csv", index=False)
    per_state = {}
    for st in states:
        idx = [i for i, ex in enumerate(test) if ex["state"] == st]
        if idx:
            _, ms = ft.evaluate([test[i] for i in idx], [resp_list[i] for i in idx],
                                horizon, bins, band)
            per_state[st] = ms
    usage = [responses[ex["id"]]["usage"] for ex in test if ex["id"] in responses]
    tok = {k: (sum(u[k] for u in usage) if usage else 0)
           for k in ("prompt_tokens", "thinking_tokens", "output_tokens")}
    fin = {}
    for ex in test:
        if ex["id"] in responses:
            k = str(responses[ex["id"]].get("finish_reason"))
            fin[k] = fin.get(k, 0) + 1
    metrics = {
        "tag": tag, **client_cfg, "zero_shot": True, "split": split,
        "input": input_desc,
        "nochange_band_relative": band, "horizon": horizon, "bins": ft.bin_names(bins),
        "n_test": len(test), "n_responded": len(usage),
        "n_parsed": m["n_parsed"],
        "direction_accuracy": m["direction_accuracy"],
        "bin_exact_accuracy_moves": m["bin_exact_accuracy_moves"],
        "bin_within1_accuracy_moves": m["bin_within1_accuracy_moves"],
        "operator_accuracy": m["operator_accuracy"],
        "units_rough_mae": ("hospital admissions (operator applied in 100x-z index units, "
                            "back-transformed via (v/100)*sigma+mu)"
                            if any(ex.get("denorm_sigma") is not None for ex in test)
                            else "counts (raw hospitalizations; baseline × predicted-bin lower edge)"),
        "rough_MAE_counts": m["rough_MAE_counts"], "rough_MSE_counts": m["rough_MSE_counts"],
        "baseline_MAE_counts": m["baseline_MAE_counts"], "baseline_MSE_counts": m["baseline_MSE_counts"],
        "improvement_MAE_vs_baseline": m["improvement_MAE_vs_baseline"],
        "per_state": per_state,
        "token_totals": tok,
        "mean_thinking_tokens_per_call": (tok["thinking_tokens"] / len(usage)) if usage else None,
        "finish_reasons": fin,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[{tag}] direction_acc={m['direction_accuracy']:.3f}  "
          f"bin_exact(moves)={m['bin_exact_accuracy_moves']:.3f}  "
          f"operator_acc={m['operator_accuracy']:.3f}  | rough MAE={m['rough_MAE_counts']:.2f} "
          f"vs baseline {m['baseline_MAE_counts']:.2f} ({m['improvement_MAE_vs_baseline']:+.2f})  "
          f"(parsed {m['n_parsed']}/{len(test)}; mean thinking tokens/call "
          f"{metrics['mean_thinking_tokens_per_call']})", flush=True)
    return metrics


def write_summary(base: Path, all_metrics: List[Dict]) -> None:
    prev = {}
    sp = base / "summary.json"
    if sp.exists():
        prev = {m["tag"]: m for m in json.loads(sp.read_text())}
    for m in all_metrics:
        prev[m["tag"]] = m
    rows = list(prev.values())
    sp.write_text(json.dumps(rows, indent=2))
    cols = ["tag", "n_test", "n_parsed", "direction_accuracy", "bin_exact_accuracy_moves",
            "bin_within1_accuracy_moves", "operator_accuracy", "rough_MAE_counts",
            "baseline_MAE_counts", "improvement_MAE_vs_baseline", "mean_thinking_tokens_per_call"]
    flat = []
    for m in rows:
        r = {c: m.get(c) for c in cols}
        for st, ms in (m.get("per_state") or {}).items():
            r[f"{st}_direction_accuracy"] = ms["direction_accuracy"]
            r[f"{st}_operator_accuracy"] = ms["operator_accuracy"]
            r[f"{st}_rough_MAE_counts"] = ms["rough_MAE_counts"]
        flat.append(r)
    pd.DataFrame(flat).to_csv(base / "summary.csv", index=False)
    print(f"\nsummary → {sp}\n" + pd.DataFrame(flat)[cols].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--providers", default="gemini,gpt", help="comma list of gemini,gpt")
    p.add_argument("--data-mode", default="influenza", choices=("influenza",) + COVID_BITONG_MODES,
                   help="influenza (GA/NY/NJ corpus test split) or covid_TSFM_bitong_org (TX/VT/NJ "
                        "final-fold test block; operator task on the z26w index, ±20-point band)")
    p.add_argument("--mode", default="chronos2_full", choices=("chronos2_full",))
    p.add_argument("--split", choices=("test", "val"), default="test")
    p.add_argument("--states", default=None,
                   help="corpus dir is <STATES joined by _>-<corpus-model> (default per data mode)")
    p.add_argument("--corpus-model", default=None,
                   help="corpus tag (<STATES>-<tag>); default per data mode (see DEFAULT_CORPUS)")
    p.add_argument("--input-root", type=Path, default=ft.INPUT_ROOT)
    p.add_argument("--prompt-dir", type=Path, default=None,
                   help="default: prompts/covid for covid modes, prompts/influenza otherwise")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--text", choices=("keep", "drop"), default="keep",
                   help="keep = plain FluView reports of the last --text-weeks weeks (default); "
                        "drop = notext stub (scale_residualcot_notext input)")
    p.add_argument("--text-weeks", type=int, default=TEXT_WEEKS,
                   help="with --text keep: number of most recent report weeks to include")
    p.add_argument("--reference-test-jsonl", type=Path, default=None,
                   help="finetune run test.jsonl to verify the rebuilt prompts against "
                        "(default: per --text, see REFERENCE_RUN)")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--keep-no-think", action="store_true",
                   help="keep the Qwen '/no_think' token in the system prompt")
    p.add_argument("--gemini-model", default=GEMINI_MODEL)
    p.add_argument("--gemini-thinking-level", default=GEMINI_THINKING_LEVEL,
                   choices=("minimal", "low", "medium", "high"))
    p.add_argument("--gemini-thinking-budget", type=int, default=None,
                   help="gemini-2.5-* only: integer thinking budget (replaces thinking_level; "
                        "128 = 2.5-pro floor). Run tag becomes <model>_budget-<N>")
    p.add_argument("--gpt-model", default=GPT_MODEL)
    p.add_argument("--gpt-reasoning-effort", default=GPT_REASONING_EFFORT,
                   choices=("none", "minimal", "low", "medium", "high"))
    p.add_argument("--openai-cred", type=Path, default=OPENAI_CRED_PATH)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--max-output-tokens", type=int, default=ft.MAX_NEW_TOKENS)
    p.add_argument("--nochange-band", type=float, default=None,
                   help="default: ±10% relative (influenza) / ±20 index points (covid bitong)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="first N test rows (smoke test)")
    p.add_argument("--force", action="store_true", help="discard cached responses and re-query")
    args = p.parse_args()
    if args.prompt_dir is None:
        args.prompt_dir = P.prompt_dir_for(args.data_mode)
    return args


def main():
    args = parse_args()
    if args.states is None:
        args.states = ft.DATA_MODE_STATES[args.data_mode]
    if args.corpus_model is None:
        args.corpus_model = DEFAULT_CORPUS[args.data_mode]
    if args.nochange_band is None:          # covid: ±20 index points; influenza: ±10% relative
        args.nochange_band = (ft.NOCHANGE_ABS_INDEX_BAND if args.data_mode in ft.COVID_MODES
                              else ft.NOCHANGE_REL_BAND)
    states = [s.strip().upper() for s in args.states.split(",") if s.strip()]
    providers = [s.strip().lower() for s in args.providers.split(",") if s.strip()]
    for pv in providers:
        if pv not in ("gemini", "gpt"):
            sys.exit(f"unknown provider {pv}")

    data_dir, combo, run_dir, horizon, bins, test, prompts, sysp = build_test_prompts(args, states)
    reports = ("FluView reports" if args.data_mode == "influenza"
               else "state respiratory surveillance reports")
    errs = "recent errors + "
    input_desc = (f"history + chronos2 baseline + {errs}notext stub (no {reports})"
                  if args.text == "drop" else
                  f"history + chronos2 baseline + {errs}PLAIN {reports} of the "
                  f"last {args.text_weeks} weeks (no generated CoT)")
    print(f"  input: {input_desc}")
    print(f"=== ZERO-SHOT API inference  data={data_dir}  corpus={run_dir}  states={combo}  "
          f"horizon={horizon}  split={args.split}  n={len(test)}  band={args.nochange_band}  "
          f"providers={providers} ===")
    if not args.no_verify:
        ref = args.reference_test_jsonl or REFERENCE_RUN_BY_MODE[args.data_mode][args.text]
        if ref is None:
            print("  [verify] no reference finetune run for this input; skipping check")
        else:
            if args.split == "val" and args.reference_test_jsonl is None:
                ref = ref.with_name("val_infer.jsonl")     # same prompts, no assistant turn
            if args.text == "keep" and args.text_weeks < TEXT_WEEKS:
                print(f"  [verify] --text-weeks {args.text_weeks} < corpus window; skipping check")
            else:
                verify_against_reference(test, prompts, ref)
    if args.limit:
        test, prompts = test[:args.limit], prompts[:args.limit]
        print(f"  --limit {args.limit}: using the first {len(test)} test rows")

    variant = INPUT_SUBDIR[args.text].format(n=args.text_weeks)
    base = args.output_root / data_dir / args.mode / combo / variant
    if args.split != "test":
        base = base / args.split
    if args.limit:
        base = base / f"smoke_limit{args.limit}"
    base.mkdir(parents=True, exist_ok=True)
    with (base / "test.jsonl").open("w") as f:
        for ex, p in zip(test, prompts):
            f.write(json.dumps({"id": ex["id"], "state": ex["state"],
                                "reference_date": ex["reference_date"],
                                "messages": [{"role": "system", "content": sysp},
                                             {"role": "user", "content": p}]}) + "\n")
    print(f"  wrote {len(test)} {args.split} prompts → {base / 'test.jsonl'}")
    print(f"  system prompt: {sysp!r}")

    all_metrics = []
    for pv in providers:
        if pv == "gemini":
            client = GeminiZS(args.gemini_model, args.gemini_thinking_level,
                              args.max_output_tokens, args.temperature,
                              thinking_budget=args.gemini_thinking_budget)
        else:
            client = GPTZS(args.gpt_model, args.gpt_reasoning_effort, args.max_output_tokens,
                           args.temperature, args.openai_cred)
        tag = client.tag()
        out_dir = base / tag
        print(f"\n--- {tag} → {out_dir} ---")
        responses = run_model(client, tag, test, prompts, sysp, out_dir, args.workers, args.force)
        all_metrics.append(evaluate_model(tag, client.config(), test, responses, horizon, bins,
                                          args.nochange_band, out_dir, states, input_desc, args.split))
    write_summary(base, all_metrics)


if __name__ == "__main__":
    main()
