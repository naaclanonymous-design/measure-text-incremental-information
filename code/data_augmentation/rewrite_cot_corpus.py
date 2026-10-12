#!/usr/bin/env python3
"""Build the NO-RESIDUAL ("nores") corpus by EDITING an existing CoT corpus (source never
modified). Every training target's reasoning is rewritten so that it no longer refers to the
recent-baseline-error block; the direction / size labels, the input blocks, the bins and the
split are copied unchanged. Meant for the released COVID finetune prompt
(prompts/covid/finetune_prompt.txt, data modes covid_TSFM_bitong / covid_TSFM_bitong_org), which
shows NO recent-error item (history, baseline, reports only).

Why: on the TX/VT/NJ z26w series
the sign of the recent baseline errors does not predict the next residual (corr ≈ 0 over
all train-window origins, −0.34 on the test winter), yet 97% of the training targets cite
the error block and 56% argue "errors have been persistently positive/negative → still
too low/high". Worse, the Step-1 filter kept exactly the origins where that rule happened
to hold (corr +0.34 among kept moves vs −0.32 among dropped origins), so the finetune
learns a selection artefact. This builder removes the error-block reasoning from the
targets; the matching finetune modes remove the block from the inputs.

Per accepted.jsonl row (all splits are copied; only rows with a reasoning are edited):
  move      : step1_basis_cot + step2_size_cot  → rewritten DIRECTION + SIZE reasoning
  no_change : step1_basis_cot                    → rewritten DIRECTION reasoning
  abstain   : abstain_cot (abstain-fix corpora)  → rewritten DIRECTION reasoning
              (a row with no input-specific abstain_cot is DROPPED: its target would be
              the canned template, which finetune.py would otherwise fall back to)

One Gemini call (temperature 0, no sampling) edits the text under
code/data_augmentation/prompts/rewrite_cot.txt; the edit must then pass, in this order:
  (a) no residual reference left (regex RESID_RE — the same list the dry run reports);
  (b) ≥ 1 double-quoted phrase and EVERY double-quoted phrase found verbatim (normalised)
      in the row's weekly_text_block (the pipeline's verbatim gate);
  (c) a move row must have both a DIRECTION and a SIZE section;
  (d) unless --no-judge: the Flash faithfulness judge (semantic, as in step1_filter.py) and
      the NO-RESIDUAL grounding judge (rewrite_grounding_flash.txt, stated direction = the
      row's fixed direction) with score ≥ --min-grounding.
A failed edit is retried ONCE with the failure reasons appended to the prompt; a second
failure, or a model "DROP" verdict, drops the row (never keeps the original text).

Outputs (in --dst, default <src>-nores → finetune.py --corpus-model <tag>-nores):
  accepted.jsonl            kept rows; edited fields replaced, originals kept as *_orig,
                            plus rewrite_status / rewrite_attempt / rewrite_checks
  dropped.jsonl             dropped rows (with the reason)
  rewrite_decisions.jsonl   one line per source row (kind, attempts, checks, decision)
  rewrite_report.json       counts by class / state / failure reason + dry-run statistics
  rewrite_generated.jsonl   resume cache of the raw model edits (id, attempt)
  rewrite_judged.jsonl      resume cache of the judge verdicts (id, attempt)
  bins.json                 copied from the source

Usage:
  export GOOGLE_CLOUD_PROJECT=...            # Vertex ADC, as for the pipeline
  python rewrite_cot_corpus.py --dry-run                 # statistics only, no API calls
  python rewrite_cot_corpus.py --limit 3                 # live smoke test on 3 rows
  python rewrite_cot_corpus.py                           # full run (resumable)
  python finetune.py --data-mode covid_TSFM_bitong --corpus-model gemini-abstainfix-nores ...

The released data/cot_corpora/covid/TX_VT_NJ-gemini-final/ is exactly this script's output
(257 rows) for the abstain-fix corpus of the pipeline run behind the paper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_CODE_ROOT = Path(__file__).resolve().parents[1]          # <paper-content>/code
for _d in ("cot_generation", "finetune", "zero_shot"):
    sys.path.insert(0, str(_CODE_ROOT / _d))
sys.path.insert(0, str(_CODE_ROOT))                         # paper_paths.py
import paper_paths as P                                     # noqa: E402
AUG_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"   # the prompts of this stage
import cot_common as C
from cot_common import logger

MODE = "chronos2_full"
HISTORY_LEN = 12
DST_SUFFIX = "-nores"
# per data mode: source corpus, prompt dir (rewrite_cot.txt / rewrite_*_flash.txt, flat),
# horizon (number of predicted weeks = direction/size values per conclusion line) and the
# no-change band as cot_common renders it. Only the COVID profile is part of the release.
PROFILES = {
    "covid_TSFM_updated": dict(
        src=P.COT_OUTPUT_DIR / "covid_TSFM_updated" / "chronos2_full" / "TX_VT_NJ-gemini-abstainfix",
        prompt_dir=AUG_PROMPT_DIR, subdir="", horizon=1, band=20.0),
}
PROF: dict = {}          # the active profile (set in main from --data-mode)
DATA_MODE = "covid_TSFM_updated"

# ── residual-reference detector (gate (a); also the dry-run statistic) ────────────────
RESID_PATTERNS = [
    r"\bresidual",
    r"baseline[- ]error", r"recent[- ]error", r"forecast[- ]error", r"error[- ]pattern",
    r"far[- ]horizon", r"3[- ]week[- ]ahead", r"three[- ]week[- ]ahead",
    r"\berrors?\b[^.]{0,40}\b(index points|positive|negative|band)",
    r"\bunder[- ]?predict", r"\bover[- ]?predict", r"\bunder[- ]?forecast", r"\bover[- ]?forecast",
    r"\bunderestimat", r"\boverestimat",
    r"\btoo (low|high)\b", r"\bcalibrat",
    r"\b(has|have|had) been (consistently|persistently|chronically|repeatedly)",
    r"\bmiss(ed|es|ing)? (the|its) (actual|true|realized)",
    # second pass (2026-09-27): wordings the first run let through
    # ("the baseline is overshooting the coming level" / "a medium size avoids over-correcting"
    #  are legitimate baseline-vs-phase / size-choice statements and are NOT flagged)
    r"\b(predictive|prediction|forecasting|past|previous|prior|recent|model'?s?|baseline'?s?) "
    r"(errors?|misses|inaccurac)",
    r"\b(past|previous|prior|recent|earlier) (overshoot|undershoot|over-?correction|under-?correction)",
    r"\berror (history|pattern|signal|record|block)",
    # influenza wording (two-horizon percentage block)
    r"\b(1|one|next)[- ]week[- ]ahead", r"\bnext[- ]week (error|residual|miss|forecast error)",
    r"\bnear[- ]horizon", r"\b(both|two) horizons?\b", r"\bpercentage (error|miss|residual)",
]
RESID_RE = re.compile("|".join(RESID_PATTERNS), re.IGNORECASE)
# double-quoted spans only (cot_common's extractor also opens on apostrophes)
_DQUOTE_RE = re.compile(r"[\"“]([^\"“”]{3,})[\"”]")
_SECTION_RE = re.compile(r"===\s*DIRECTION REASONING\s*===(.*?)(?:===\s*SIZE REASONING\s*===(.*))?$",
                         re.IGNORECASE | re.DOTALL)
_TAG_LINE_RE = re.compile(r"^\s*(Directions?|Sizes?|Reason|Direction|Size)\s*:", re.IGNORECASE)
DIR_WORD = {1: "increase", -1: "decrease", 0: "no change"}


# ══════════════════════════════════════════════════════════════════════════════
# Small helpers
# ══════════════════════════════════════════════════════════════════════════════
def read_jsonl(path: Path) -> List[dict]:
    return list(C.iter_jsonl(path)) if path.exists() else []


def strip_tail_lines(cot: str) -> str:
    """Drop trailing Directions: / Sizes: / Reason: bookkeeping lines."""
    lines = (cot or "").strip().splitlines()
    while lines and _TAG_LINE_RE.match(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def strip_tag_lines_anywhere(text: str) -> str:
    """The edit must not carry conclusion lines at all (finetune.py adds them)."""
    return "\n".join(l for l in (text or "").splitlines() if not _TAG_LINE_RE.match(l)).strip()


def row_class(r: dict) -> str:
    if r.get("operator") == "no-correction" or r.get("date_status") == "abstain":
        return "abstain"
    return "move" if r.get("operator") == "correction" else "no_change"


def fixed_directions(r: dict, cls: str) -> List[int]:
    """One direction per predicted week (horizon = PROF['horizon'])."""
    H = PROF.get("horizon", 1)
    if cls == "abstain":
        return [0] * H
    bd = list(r.get("basis_directions") or [])
    return [int(v) for v in (bd + [0] * H)[:H]]


def fixed_direction(r: dict, cls: str) -> int:
    """The first predicted week's direction (used for single-value bookkeeping)."""
    return fixed_directions(r, cls)[0]


def directions_words(r: dict, cls: str) -> str:
    return ", ".join(DIR_WORD[d] for d in fixed_directions(r, cls))


def conclusion_line(r: dict, cls: str) -> str:
    H = PROF.get("horizon", 1)
    d = directions_words(r, cls)
    if cls == "move":
        sizes = [s.strip() for s in (r.get("size_names") or "").split(",") if s.strip()]
        sizes = (sizes + ["none"] * H)[:H]
        return f"Direction: {d}\nSize: {', '.join(sizes)}"
    return f"Direction: {d}\nSize: {', '.join(['none'] * H)}"


CANNED = "CANNED"   # marker: abstain row whose target would be the canned template


def original_parts(r: dict, cls: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(direction_text, size_text, drop_reason)."""
    if cls == "abstain":
        a = strip_tail_lines(r.get("abstain_cot") or "")
        if not a:
            return None, None, CANNED
        return a, None, None
    s1 = strip_tail_lines(r.get("step1_basis_cot") or "")
    if not s1:
        return None, None, "no step1_basis_cot"
    if cls == "move":
        s2 = strip_tail_lines(r.get("step2_size_cot") or "")
        return s1, (s2 or None), None
    return s1, None, None


def resid_hits(text: str) -> List[str]:
    return sorted({m.group(0).strip() for m in RESID_RE.finditer(text or "")})


def resid_sentence_share(text: str) -> Tuple[int, int]:
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text or "") if len(s.strip()) > 20]
    return sum(bool(RESID_RE.search(s)) for s in sents), len(sents)


def dquote_check(text: str, weekly_text_block: str) -> Dict:
    norm_text = C.assembled_text_for_faithfulness(weekly_text_block)
    quotes = _DQUOTE_RE.findall(text or "")
    missing = [q for q in quotes if re.sub(r"\s+", " ", q.lower()).strip() not in norm_text]
    return {"n_quotes": len(quotes), "n_found": len(quotes) - len(missing),
            "missing": missing, "faithful": len(quotes) >= 1 and not missing}


def parse_edit(raw: str, cls: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(direction_text, size_text, error). 'DROP' → (None, None, 'DROP')."""
    txt = (raw or "").strip()
    if not txt:
        return None, None, "empty model output"
    if re.match(r"^\W*DROP\b", txt, re.IGNORECASE) and len(txt) < 40:
        return None, None, "DROP"
    # a reply cut off by the output budget (thinking tokens count against it) ends mid-sentence
    if txt[-1] not in ".!?\"”)’'":
        return None, None, ("the output was cut off mid-sentence (write the complete edited "
                            "reasoning; keep it no longer than the original)")
    m = _SECTION_RE.search(txt)
    if not m:
        return None, None, "output lacks the '=== DIRECTION REASONING ===' section"
    d = strip_tag_lines_anywhere(m.group(1))
    s = strip_tag_lines_anywhere(m.group(2) or "")
    if not d:
        return None, None, "empty direction reasoning"
    if cls == "move" and not s:
        return None, None, "move row without a SIZE section"
    if cls != "move":
        s = ""
    return d, (s or None), None


# ══════════════════════════════════════════════════════════════════════════════
# Prompt rendering
# ══════════════════════════════════════════════════════════════════════════════
def render_rewrite(r: dict, cls: str, d_text: str, s_text: Optional[str], tmpl: str,
                   feedback: Optional[List[str]]) -> str:
    original = d_text if not s_text else f"{d_text}\n\n[SIZE PART]\n{s_text}"
    fb = ""
    if feedback:
        fb = ("\nYOUR PREVIOUS EDIT WAS REJECTED for these reasons — fix ALL of them:\n"
              + "\n".join(f"  • {x}" for x in feedback) + "\n")
    kind = {"move": "move (the analyst revises the baseline; direction AND size are fixed)",
            "no_change": "no change (the baseline is kept; the reasoning explains why the reports "
                         "do not warrant a revision beyond the band)",
            "abstain": "abstain / keep baseline (the reasoning explains why the reports do not "
                       "support a revision)"}[cls]
    return C.fill_template(tmpl, {
        "history_len": HISTORY_LEN,
        "horizon_len": PROF.get("horizon", 1),
        "nochange_band": C.fmt_nochange_band(MODE, PROF.get("band", 20.0), DATA_MODE),
        "history_block": r["history_block"],
        "baseline_block": r["baseline_block"],
        "weekly_text_block": r["weekly_text_block"],
        "record_kind": kind,
        "conclusion_line": conclusion_line(r, cls),
        "original_reasoning": original,
        "feedback_block": fb,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Checks and judges
# ══════════════════════════════════════════════════════════════════════════════
def offline_checks(r: dict, cls: str, d: str, s: Optional[str]) -> Tuple[Dict, List[str]]:
    combined = d if not s else f"{d}\n\n{s}"
    res: Dict = {}
    fails: List[str] = []
    hits = resid_hits(combined)
    res["resid_hits"] = hits
    if hits:
        fails.append("the edited text still refers to baseline errors / residuals: "
                     + "; ".join(f'"{h}"' for h in hits[:6]))
    q = dquote_check(combined, r["weekly_text_block"])
    # the source corpus was gated by the SEMANTIC Flash judge, so some original quotes are
    # paraphrases that fail a verbatim check; the edit may keep those (rule 2 says keep every
    # quote) but must not introduce NEW unverifiable quotes. The semantic judge then rules.
    orig = (r.get("step1_basis_cot") or "") + "\n" + (r.get("step2_size_cot") or "") \
           + "\n" + (r.get("abstain_cot") or "")
    orig_missing = {re.sub(r"\s+", " ", m.lower()).strip()
                    for m in dquote_check(orig, r["weekly_text_block"])["missing"]}
    new_missing = [m for m in q["missing"]
                   if re.sub(r"\s+", " ", m.lower()).strip() not in orig_missing]
    res.update({"n_quotes": q["n_quotes"], "n_quotes_found": q["n_found"],
                "missing_quotes": q["missing"], "new_missing_quotes": new_missing})
    if q["n_quotes"] < 1:
        fails.append("no double-quoted phrase from the report text")
    elif new_missing:
        fails.append("these quoted phrases are not found verbatim in the report text: "
                     + "; ".join(f'"{m}"' for m in new_missing[:6]))
    if cls == "move" and s and "evidence:" in (r.get("step2_size_cot") or "").lower() \
            and "evidence:" not in s.lower():
        res["size_evidence_line_lost"] = True   # informational only
    return res, fails


def judge(llm, r: dict, cls: str, d: str, s: Optional[str], args, tmpl_f: str,
          tmpl_g: str) -> Tuple[Dict, List[str]]:
    combined = d if not s else f"{d}\n\n{s}"
    res: Dict = {"flash_faithful": None, "grounding_score": None}
    fails: List[str] = []
    # faithfulness: a quote found VERBATIM in the report is faithful by construction, so the
    # semantic judge only rules when some (pre-existing, paraphrased) quote is not verbatim
    q = dquote_check(combined, r["weekly_text_block"])
    if q["missing"]:
        try:
            raw = llm.generate(model=args.flash_model,
                               prompt=C.fill_template(tmpl_f, {"weekly_text_block": r["weekly_text_block"],
                                                               "cot": combined}),
                               temperature=0.0, top_p=1.0, max_output_tokens=args.judge_max_tokens)
            res["flash_faithful"] = C.parse_faithful_verdict(raw)
        except Exception as e:                       # noqa: BLE001
            logger.error("faithfulness judge failed %s: %s", r["id"], e)
        if res["flash_faithful"] is False:
            fails.append("the faithfulness judge found a quoted phrase unsupported by the report")
            return res, fails
    else:
        res["flash_faithful"] = "verbatim"
    try:
        raw = llm.generate(model=args.grounding_model,
                           prompt=C.fill_template(tmpl_g, {
                               "history_block": r["history_block"],
                               "baseline_block": r["baseline_block"],
                               "weekly_text_block": r["weekly_text_block"],
                               "directions": directions_words(r, cls),
                               "horizon_len": PROF.get("horizon", 1),
                               "cot": combined}),
                           temperature=0.0, top_p=1.0, max_output_tokens=args.judge_max_tokens)
        res["grounding_score"] = C.parse_judge_score(raw)
    except Exception as e:                       # noqa: BLE001
        logger.error("grounding judge failed %s: %s", r["id"], e)
    if res["grounding_score"] is None or res["grounding_score"] < args.min_grounding:
        fails.append(f"the grounding judge scored the argument {res['grounding_score']} "
                     f"(< {args.min_grounding}): the stated direction is not sufficiently "
                     f"supported by the report text and value history alone")
    return res, fails


# ══════════════════════════════════════════════════════════════════════════════
# Resume caches
# ══════════════════════════════════════════════════════════════════════════════
class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.rows: Dict[Tuple[str, int], dict] = {}
        for row in read_jsonl(path):
            self.rows[(row["id"], int(row["attempt"]))] = row

    def get(self, rid: str, attempt: int) -> Optional[dict]:
        return self.rows.get((rid, attempt))

    def put(self, row: dict) -> None:
        with self.lock:
            self.rows[(row["id"], int(row["attempt"]))] = row
            with self.path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# One row
# ══════════════════════════════════════════════════════════════════════════════
def process_row(r: dict, args, llm, judge_llm, tmpl_rw: str, tmpl_f: str, tmpl_g: str,
                gen_cache: Cache, jud_cache: Cache) -> Tuple[dict, Optional[dict]]:
    """Returns (decision, edited_row_or_None)."""
    cls = row_class(r)
    d0, s0, why = original_parts(r, cls)
    dec = {"id": r["id"], "state": r.get("state"), "reference_date": r.get("reference_date"),
           "split": r.get("split"), "class": cls, "direction": directions_words(r, cls),
           "orig_resid_hits": resid_hits((d0 or "") + "\n" + (s0 or "")), "attempts": []}
    if r.get("split") not in args.rewrite_splits:
        # e.g. influenza test rows: their reasoning is never a training/eval target — copy
        dec.update(decision="kept", reason="split not rewritten (copied unchanged)")
        return dec, {**r, "rewrite_status": "copied"}
    if why == CANNED:
        if args.keep_canned_abstain:
            # finetune.py falls back to the data mode's canned text (the COVID modes use a
            # canned text without the error-pattern sentence), so the row can stay
            dec.update(decision="kept", reason="canned abstain kept (no abstain_cot)")
            return dec, {**r, "rewrite_status": "canned_kept"}
        dec.update(decision="dropped", reason="abstain row without an input-specific abstain_cot (canned template)")
        return dec, None
    if why:
        dec.update(decision="dropped", reason=why)
        return dec, None
    feedback: List[str] = []
    for attempt in range(1, args.max_attempts + 1):
        prompt = render_rewrite(r, cls, d0, s0, tmpl_rw, feedback or None)
        # cache key = prompt + generation settings (a budget/thinking change invalidates it)
        key = hashlib.sha1(f"{prompt}|{args.gen_model}|{args.gen_max_tokens}|"
                           f"{args.thinking_level}|{args.thinking_budget}".encode()).hexdigest()[:12]
        cached = gen_cache.get(r["id"], attempt)
        if cached is not None and cached.get("prompt_sha") == key:
            raw = cached["raw"]
        else:
            try:
                raw = llm.generate(model=args.gen_model, prompt=prompt, temperature=0.0,
                                   top_p=1.0, max_output_tokens=args.gen_max_tokens)
            except Exception as e:                   # noqa: BLE001
                logger.error("rewrite failed %s attempt %d: %s", r["id"], attempt, e)
                dec["attempts"].append({"attempt": attempt, "error": str(e)})
                continue
            gen_cache.put({"id": r["id"], "attempt": attempt, "prompt_sha": key,
                           "model": args.gen_model, "raw": raw})
        d, s, err = parse_edit(raw, cls)
        att = {"attempt": attempt}
        if err == "DROP":
            att["decision"] = "model DROP"
            dec["attempts"].append(att)
            dec.update(decision="dropped", reason="model judged the conclusion unsupported without the error block")
            return dec, None
        if err:
            att["fails"] = [err]
            dec["attempts"].append(att)
            feedback = [err]
            continue
        checks, fails = offline_checks(r, cls, d, s)
        att["checks"] = checks
        if not fails and not args.no_judge:
            jc = jud_cache.get(r["id"], attempt)
            if jc is not None and jc.get("prompt_sha") == key:
                jres, jfails = jc["res"], jc["fails"]
            else:
                jres, jfails = judge(judge_llm, r, cls, d, s, args, tmpl_f, tmpl_g)
                jud_cache.put({"id": r["id"], "attempt": attempt, "prompt_sha": key,
                               "res": jres, "fails": jfails})
            checks.update(jres)
            fails += jfails
        att["fails"] = fails
        dec["attempts"].append(att)
        if fails:
            feedback = fails
            continue
        # accepted: build the edited row
        out = dict(r)
        if cls == "abstain":
            out["abstain_cot_orig"] = r.get("abstain_cot")
            out["abstain_cot"] = d
            if r.get("step1_basis_cot"):
                out["step1_basis_cot_orig"] = r.get("step1_basis_cot")
                out["step1_basis_cot"] = d
        else:
            out["step1_basis_cot_orig"] = r.get("step1_basis_cot")
            out["step1_basis_cot"] = d
            if cls == "move":
                out["step2_size_cot_orig"] = r.get("step2_size_cot")
                out["step2_size_cot"] = s or ""
        out["rewrite_status"] = "rewritten"
        out["rewrite_attempt"] = attempt
        out["rewrite_checks"] = checks
        dec.update(decision="kept", attempt=attempt)
        return dec, out
    dec.update(decision="dropped", reason="no accepted edit after "
               f"{args.max_attempts} attempt(s): " + "; ".join(feedback[:3]))
    return dec, None


# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-mode", choices=sorted(PROFILES), default="covid_TSFM_updated",
                    help="selects the default corpus / prompt set / horizon / band (PROFILES)")
    ap.add_argument("--src", type=Path, default=None, help="source corpus dir (default: profile)")
    ap.add_argument("--dst", type=Path, default=None,
                    help=f"output dir (default <src>{DST_SUFFIX} → --corpus-model <tag>{DST_SUFFIX})")
    ap.add_argument("--prompt-dir", type=Path, default=None, help="default: profile")
    ap.add_argument("--subdir", type=str, default=None,
                    help="optional subdir of --prompt-dir holding rewrite_cot.txt / rewrite_*_flash.txt (default: flat)")
    ap.add_argument("--rewrite-splits", type=str, default="train,val",
                    help="rows of other splits are copied unchanged (their reasoning is never a target)")
    ap.add_argument("--keep-canned-abstain", action="store_true",
                    help="keep abstain rows that have no abstain_cot (finetune.py's nores canned "
                         "fallback is used) instead of dropping them — needed for corpora without "
                         "an abstain-fix pass, e.g. influenza")
    ap.add_argument("--gen-model", type=str, default=C.PRO_MODEL)
    ap.add_argument("--flash-model", type=str, default=C.FLASH_MODEL, help="faithfulness judge")
    ap.add_argument("--grounding-model", type=str, default=C.FLASH_MODEL, help="grounding judge")
    ap.add_argument("--min-grounding", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=2, help="1 edit + 1 retry with feedback")
    # gemini-3 thinking tokens count against max_output_tokens: keep thinking LOW and the
    # budget large, otherwise the edit is cut off mid-sentence (seen at 4096 / default level)
    ap.add_argument("--gen-max-tokens", type=int, default=16384)
    ap.add_argument("--judge-max-tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--thinking-budget", type=int, default=None)
    ap.add_argument("--thinking-level", type=str, default="low", choices=["low", "high"])
    ap.add_argument("--no-judge", action="store_true", help="offline checks only (no judge calls)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    ap.add_argument("--ids", type=str, default=None, help="comma list of row ids to process")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API calls: report residual-reference / quote statistics only")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    C.setup_logging(args.verbose)
    global PROF, DATA_MODE
    PROF = PROFILES[args.data_mode]
    DATA_MODE = args.data_mode
    args.src = args.src or PROF["src"]
    args.prompt_dir = args.prompt_dir or PROF["prompt_dir"]
    args.subdir = args.subdir if args.subdir is not None else PROF["subdir"]
    args.rewrite_splits = {s.strip() for s in args.rewrite_splits.split(",") if s.strip()}
    print(f"data-mode {args.data_mode}: horizon={PROF['horizon']} band="
          f"{C.fmt_nochange_band(MODE, PROF['band'], DATA_MODE)} prompts={args.prompt_dir}/{args.subdir} "
          f"rewrite splits={sorted(args.rewrite_splits)} keep canned abstain={args.keep_canned_abstain}")

    src = args.src
    dst = args.dst or src.parent / f"{src.name}{DST_SUFFIX}"
    if dst.resolve() == src.resolve():
        sys.exit("--dst must differ from --src (the source corpus is never modified)")
    rows = read_jsonl(src / "accepted.jsonl")
    if not rows:
        sys.exit(f"no accepted.jsonl under {src}")
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        rows = [r for r in rows if r["id"] in want]
    if args.limit:
        rows = rows[:args.limit]
    print(f"source : {src}\noutput : {dst}\nrows   : {len(rows)}")

    # ── dry-run statistics (always printed) ────────────────────────────────────
    stats: Counter = Counter()
    per_state: Dict[str, Counter] = defaultdict(Counter)
    hit_counter: Counter = Counter()
    shares: List[float] = []
    for r in rows:
        cls = row_class(r)
        d0, s0, why = original_parts(r, cls)
        stats[f"rows_{cls}"] += 1
        per_state[r.get("state", "?")][cls] += 1
        if why:
            stats[f"no_reasoning_{cls}"] += 1
            continue
        txt = (d0 or "") + "\n" + (s0 or "")
        hits = resid_hits(txt)
        if hits:
            stats[f"resid_ref_{cls}"] += 1
            for h in hits:
                hit_counter[h.lower()] += 1
        k, n = resid_sentence_share(txt)
        if n:
            shares.append(k / n)
        q = dquote_check(txt, r["weekly_text_block"])
        if not q["faithful"]:
            stats[f"orig_quote_fail_{cls}"] += 1
    n = len(rows)
    n_ref = sum(v for k, v in stats.items() if k.startswith("resid_ref_"))
    print(f"\n=== source statistics: {n} rows; {n_ref} ({100*n_ref/max(n,1):.0f}%) reference the "
          f"error block; median share of sentences referencing it = "
          f"{sorted(shares)[len(shares)//2] if shares else 0:.2f}")
    for cls in ("move", "no_change", "abstain"):
        print(f"   {cls:10s} rows={stats[f'rows_{cls}']:3d}  with error-block refs="
              f"{stats[f'resid_ref_{cls}']:3d}  no reasoning={stats[f'no_reasoning_{cls}']:2d}  "
              f"original quote check fails={stats[f'orig_quote_fail_{cls}']:2d}")
    print("   most frequent matches:", ", ".join(f"{k} ({v})" for k, v in hit_counter.most_common(12)))
    dry = {"n_rows": n, "n_with_resid_refs": n_ref, "stats": dict(stats),
           "per_state": {k: dict(v) for k, v in per_state.items()},
           "top_matches": hit_counter.most_common(40)}
    if args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "rewrite_report_dryrun.json").write_text(json.dumps(dry, indent=2))
        print(f"\n[dry-run] no API calls; statistics → {dst / 'rewrite_report_dryrun.json'}")
        return

    # ── live run ───────────────────────────────────────────────────────────────
    dst.mkdir(parents=True, exist_ok=True)
    tmpl_rw = C.load_prompt("rewrite_cot.txt", args.prompt_dir, args.subdir)
    tmpl_f = C.load_prompt("rewrite_faithfulness_flash.txt", args.prompt_dir, args.subdir)
    tmpl_g = C.load_prompt("rewrite_grounding_flash.txt", args.prompt_dir, args.subdir)
    llm = C.make_llm_client("gemini", thinking_budget=args.thinking_budget,
                            thinking_level=args.thinking_level)
    judge_llm = C.make_llm_client("gemini", thinking_budget=0)   # judges: thinking OFF
    gen_cache = Cache(dst / "rewrite_generated.jsonl")
    jud_cache = Cache(dst / "rewrite_judged.jsonl")

    decisions: List[dict] = []
    kept: List[dict] = []
    dropped: List[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_row, r, args, llm, judge_llm, tmpl_rw, tmpl_f, tmpl_g,
                          gen_cache, jud_cache): r for r in rows}
        done = 0
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                dec, out = fut.result()
            except Exception as e:                   # noqa: BLE001
                logger.error("row %s crashed: %s", r["id"], e)
                dec, out = {"id": r["id"], "class": row_class(r), "decision": "dropped",
                            "reason": f"crash: {e}", "attempts": []}, None
            decisions.append(dec)
            if out is not None:
                kept.append(out)
            else:
                dropped.append({**r, "rewrite_status": "dropped", "rewrite_reason": dec.get("reason")})
            done += 1
            if done % 10 == 0 or done == len(rows):
                nk = len(kept)
                logger.info("progress %d/%d  kept=%d dropped=%d", done, len(rows), nk, done - nk)

    # subset run (--ids / --limit) on an existing output: MERGE into the previous
    # accepted / dropped / decisions instead of overwriting them with the subset
    subset = bool(args.ids or args.limit)
    if subset and (dst / "accepted.jsonl").exists():
        done_ids = {r["id"] for r in rows}
        prev_kept = [r for r in read_jsonl(dst / "accepted.jsonl") if r["id"] not in done_ids]
        prev_drop = [r for r in read_jsonl(dst / "dropped.jsonl") if r["id"] not in done_ids]
        prev_dec = [d for d in read_jsonl(dst / "rewrite_decisions.jsonl") if d["id"] not in done_ids]
        kept, dropped, decisions = prev_kept + kept, prev_drop + dropped, prev_dec + decisions
        print(f"[merge] subset of {len(done_ids)} rows merged into the existing output "
              f"({len(prev_kept)} kept / {len(prev_drop)} dropped rows carried over)")
        rows = read_jsonl(src / "accepted.jsonl")     # source order for sorting
    order = {r["id"]: i for i, r in enumerate(rows)}
    kept.sort(key=lambda r: order[r["id"]])
    dropped.sort(key=lambda r: order[r["id"]])
    decisions.sort(key=lambda d: order[d["id"]])
    with (dst / "accepted.jsonl").open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (dst / "dropped.jsonl").open("w") as f:
        for r in dropped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (dst / "rewrite_decisions.jsonl").open("w") as f:
        for d in decisions:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    if (src / "bins.json").exists():
        shutil.copy(src / "bins.json", dst / "bins.json")

    # ── report ─────────────────────────────────────────────────────────────────
    rep: Dict = {"source": str(src), "n_source": len(rows), "n_kept": len(kept),
                 "n_dropped": len(dropped), "gen_model": args.gen_model,
                 "judges": None if args.no_judge else {"faithfulness": args.flash_model,
                                                       "grounding": args.grounding_model,
                                                       "min_grounding": args.min_grounding},
                 "by_class": {}, "by_state": {}, "drop_reasons": Counter(), "attempts": Counter(),
                 "residual_refs_left": 0, "dry_run_stats": dry}
    for d in decisions:
        c = rep["by_class"].setdefault(d["class"], Counter())
        c[d["decision"]] += 1
        s = rep["by_state"].setdefault(d.get("state") or "?", Counter())
        s[f"{d['class']}_{d['decision']}"] += 1
        if d["decision"] == "dropped":
            key = (d.get("reason") or "?")
            key = ("model DROP" if "model judged" in key else
                   "canned abstain" if "canned" in key else
                   "residual ref left" if "still refers" in key else
                   "quote verbatim" if "verbatim" in key or "double-quoted" in key else
                   "faithfulness judge" if "faithfulness judge" in key else
                   "grounding judge" if "grounding judge" in key else key[:60])
            rep["drop_reasons"][key] += 1
        else:
            rep["attempts"][str(d.get("attempt"))] += 1
    for r in kept:
        combined = (r.get("step1_basis_cot") or "") + "\n" + (r.get("step2_size_cot") or "") \
                   + "\n" + (r.get("abstain_cot") or "")
        if resid_hits(combined):
            rep["residual_refs_left"] += 1
    rep["by_class"] = {k: dict(v) for k, v in rep["by_class"].items()}
    rep["by_state"] = {k: dict(v) for k, v in rep["by_state"].items()}
    rep["drop_reasons"] = dict(rep["drop_reasons"])
    rep["attempts"] = dict(rep["attempts"])
    (dst / "rewrite_report.json").write_text(json.dumps(rep, indent=2))
    print(f"\n=== done: kept {len(kept)} / dropped {len(dropped)} of {len(rows)}")
    print("   by class :", rep["by_class"])
    print("   dropped  :", rep["drop_reasons"])
    print("   attempts :", rep["attempts"], " residual refs left in kept rows:", rep["residual_refs_left"])
    print(f"   corpus   → {dst / 'accepted.jsonl'}   (finetune.py --corpus-model "
          f"{dst.name.split('-', 1)[1] if '-' in dst.name else dst.name})")


if __name__ == "__main__":
    main()
