#!/usr/bin/env python3
"""Build the ABSTAIN-FIX corpus for the TX/VT/NJ covid_TSFM_updated CoT run (source corpus
never modified). Starts from the plain pipeline output (one row per origin).

Why: with 169 of 358 rows carrying ONE identical canned abstain reasoning, the finetuned
model collapses to that template ("no change" for every input). This builder gives every
kept abstain row a real, input-specific reasoning and drops the abstains whose label
contradicts the evidence — so no target string is ever repeated.

The 169 abstain origins are classified from their rejected Step-1 candidates
(step1_filtered_merged.jsonl flags + step1_generated*.jsonl texts; "faithful" = passed
the verbatim quote check — the LLM judges never ran on abstained dates):

  group 1  every faithful candidate concluded "no change" (the series then moved anyway)
           -> KEPT, no generation: the best candidate's own reasoning becomes the row's
              reasoning (it already argues from the inputs why the baseline should stand);
              ranked by quotes found, then round/candidate order. --judge-group1 optionally
              runs the Flash faithfulness judge on the chosen text and falls back to the
              next candidate if it fails.
  group 2  the candidates argued for a move but the truth stayed within the ±20 band
           -> REGENERATED as a grounded hard negative: Gemini gets the SAME four input
              blocks the finetuned model sees (never the truth) plus the strongest move
              argument, and must explain — using ONLY admissible, checkable reasons — why
              the baseline should nonetheless be kept:
                 baseline-already-reflects : the baseline forecast already prices in the
                                             trend the reports describe
                 residuals-contradict      : the recent far-horizon errors point the other
                                             way / are inside the band
                 reports-weak              : the reports conflict, are stale, or carry
                                             explicit caveats
              Each of --k candidates must (a) end "Directions: no change", (b) contain at
              least one double-quoted phrase, (c) pass the Flash faithfulness judge (the
              semantic verdict decides, as in step1_filter.py; the verbatim check of the
              double-quoted spans is recorded and is the fallback when the judge is
              unavailable), (d) score
              >= --min-grounding on the grounding judge (stated direction "no change"),
              (e) name >= 1 admissible reason on its "Reason:" line, (f) not reuse the
              canned template. Best passer = highest grounding score, then most quotes.
              No passer after --k -> another round of --k candidates is generated for
              that origin only, up to --max-k in total (default 6); still no passer -> the
              origin is DROPPED (never the template). Every candidate's judge outcome is
              kept in group2_judged.jsonl and summarized per origin in the decisions file.
  group 3  the faithful candidates were unanimous for a move and the truth moved
           (wrong way, or right way but the date still failed) -> DROPPED
  group 4  mixed / no faithful candidate / truth moved -> DROPPED

Move and no-change rows (all splits) are copied unchanged. The reasoning is stored in the
row's `abstain_cot` AND `step1_basis_cot` (basis_directions [0]); finetune.py's loader
uses `abstain_cot` for abstain rows when present and falls back to the canned text only
when absent. Rows also carry abstain_group / abstain_source / hard_negative /
abstain_reason_types for bookkeeping.

Outputs (in --dst, default <src>-abstainfix -> finetune.py --corpus-model gemini-abstainfix):
  accepted.jsonl, bins.json (copied), abstain_fix_decisions.jsonl (one line per abstain
  origin: group, decision, chosen candidate / judge results), abstain_fix_report.json,
  group2_generated.jsonl + group2_judged.jsonl (resume caches for the API stage).

Usage:
  export GOOGLE_CLOUD_PROJECT=...            # Vertex ADC, as for the pipeline
  python build_abstain_fix_corpus.py --dry-run          # classify + group 1 only, no API
  python build_abstain_fix_corpus.py                    # + group 2 regeneration (36 origins)
  python rewrite_cot_corpus.py ...        # then the "nores" rewrite -> <src>-abstainfix-nores
                                          # (= the released data/cot_corpora/covid/TX_VT_NJ-gemini-final)

Input: the full 6-stage pipeline run (code/cot_generation/run_cot_pipeline.sh covid): accepted.jsonl,
step1_filtered_merged.jsonl and step1_generated*.jsonl are all needed (the released
data/cot_corpora/covid/TX_VT_NJ-gemini/ holds only accepted.jsonl). The group-2 prompt is
code/data_augmentation/prompts/abstain_hardneg.txt.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
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

SRC_DEFAULT = P.COT_OUTPUT_DIR / "covid_TSFM_updated" / "chronos2_full" / "TX_VT_NJ-gemini"   # the 6-stage run
PROMPT_DIR_DEFAULT = P.PROMPT_DIR_COVID      # the pipeline's judge prompts (step1_faithfulness_flash / step1_grounding_pro)
MODE, DATA_MODE = "chronos2_full", "covid_TSFM_updated"
BAND = 20.0
HORIZON = 1
GROUP2_PROMPT_NAME = "abstain_hardneg.txt"
REASON_TYPES = ("baseline-already-reflects", "residuals-contradict", "reports-weak")
# the canned finetune abstain text — must never appear in a regenerated reasoning
TEMPLATE_MARK = "do not provide clear, consistent evidence for which way the baseline"
_REASON_RE = re.compile(r"Reason\s*:\s*([^\n]*)", re.IGNORECASE)
# quotes = DOUBLE-quoted spans only. cot_common's extractor also opens a "quote" at an
# apostrophe, so possessives ("colleague's ... "phrase"") produce bogus spans that fail the
# verbatim check; the group-2 prompt talks about a colleague, so this matters here.
_DQUOTE_RE = re.compile(r"[\"“]([^\"“”]{3,})[\"”]")


def dquote_faithfulness(cot: str, weekly_text_block: str) -> Dict:
    """cot_common.faithfulness restricted to double-quoted spans."""
    norm_text = C.assembled_text_for_faithfulness(weekly_text_block)
    quotes = _DQUOTE_RE.findall(cot or "")
    missing = [q for q in quotes if re.sub(r"\s+", " ", q.lower()).strip() not in norm_text]
    return {"n_quotes": len(quotes), "n_found": len(quotes) - len(missing),
            "missing": missing, "faithful": len(quotes) >= 1 and not missing}

# ══════════════════════════════════════════════════════════════════════════════
# Group-2 generation prompt (written into the prompt set once, then editable there)
# ══════════════════════════════════════════════════════════════════════════════
GROUP2_PROMPT_DEFAULT = """You are a senior time-series forecasting analyst. Every quantity below is one observed
value of the target time series for a given period (the numeric quantity being forecast, in the
normalized index units defined next).

UNITS — READ THIS FIRST. The series is NOT a raw hospitalization count. Every number
below — the value history, the baseline forecast, the recent baseline errors, and the ±{nochange_band}
no-change band — is a NORMALIZED ACTIVITY INDEX of weekly COVID-19 hospital admissions for
one US state: each week's admission count is standardized against the trailing 26-WEEK
window ending at that week (z-score = (count − 26-week mean) / 26-week standard deviation)
and then MULTIPLIED BY 100. Therefore 0 = exactly the average level of the preceding
half-year, +100 = one standard deviation above that recent average, −100 = one standard
deviation below it, and one index point = 0.01 standard deviation. NEGATIVE values are
completely normal; the index has NO floor at zero and NO fixed ceiling. Because the
normalization window trails the series, a level that merely holds steady for many weeks
drifts back toward 0 as the window catches up, and a sharp move is amplified relative to a
quiet half-year. The report text speaks in real-world quantities (counts, percentages,
levels): use it for the DIRECTION and STRENGTH of the trend, never to match numbers
against the index.

You are given, for one target series at a forecast origin (the most recent fully-observed
period):

1. {history_len}-period value history (most recent period last):
{history_block}

2. Baseline forecast for the SINGLE target — predicted week 3 (origin + 3); the nearer
   periods origin+1 and origin+2 are NOT being predicted — this is the object being
   revised or kept:
{baseline_block}

3. RECENT BASELINE ERRORS — for the last {lookback_len} OBSERVED periods, the ABSOLUTE
   residual of the 3-WEEK-AHEAD forecast, i.e. (actual − baseline forecast) in index
   points. SIGN CONVENTION: POSITIVE = the forecast was too LOW (under-predicted);
   NEGATIVE = too HIGH (over-predicted). Directly comparable to the ±{nochange_band} band:
{recent_residual_block}

4. Auxiliary text reports (the ORIGINAL report text), each labelled ONLY by how many
   periods before the forecast origin it is (oldest first; periods with no report are
   omitted). Calendar dates have been REMOVED — refer to a report ONLY by its relative
   order, NEVER by a date. This text is your PRIMARY evidence:
{weekly_text_block}

5. A colleague's argument that the baseline should be revised ({argument_direction}). It
   quotes the same reports. Read it carefully — it is the case you must answer:
{argument_block}

================================================================================
TASK — WHY THE BASELINE SHOULD BE KEPT (a grounded "no change")
================================================================================
The correct call for this origin is "no change": the warranted revision of the baseline for
predicted week 3 is within the ±{nochange_band} no-change band. Your job is to write the
reasoning an expert would give for keeping the baseline DESPITE the colleague's argument —
an honest, checkable analysis, not an excuse. Start by acknowledging, with short quotes,
the strongest evidence the colleague relies on. Then explain, from the numbers and the
reports themselves, why that evidence does not justify a move larger than {nochange_band}
index points away from the baseline. You may use ONLY these kinds of reasons, and each one
you use must be visibly tied to items 1–4:

  • baseline-already-reflects : the baseline forecast (item 2) already prices in the trend
    the reports describe — compare the history (item 1) and the baseline level/direction;
    a trend the baseline already projects is a reason to trust it, not to revise it.
  • residuals-contradict : the recent far-horizon errors (item 3) point the other way or
    sit inside the band — the same kind of forecast has recently NOT erred in the direction
    the colleague proposes.
  • reports-weak : the reports conflict with each other across periods, are stale or
    generic, or carry explicit caveats (for example reduced or voluntary reporting) that
    limit what the quoted phrases can support.

Do NOT invent facts, numbers, or report content. Do NOT mention the actual outcome — you
do not know it and must argue from the inputs alone. Do NOT dismiss the reports
wholesale: engage with the specific evidence quoted by the colleague.

--------------------------------------------------------------------------------
HOW TO QUOTE (this is checked — follow exactly)
--------------------------------------------------------------------------------
  • Quote ONLY to ground a claim in the report text (item 4). Copy a SHORT span — a few
    words, at most about 10 — as it appears in the report; one continuous phrase per quote.
  • Put in quotation marks ONLY wording that genuinely appears in the report text. NEVER
    quote your own analysis, the colleague's sentences, or any number from items 1–3.
  • All interpretation goes OUTSIDE the quotation marks, in your own words.
  • Use at least one quote and prefer a few well-chosen ones.

================================================================================
OUTPUT FORMAT
================================================================================
Return EXACTLY three parts, nothing else:
  1. ONE reasoning paragraph (no headings, bullets, markdown, or numbers stated as a
     forecast), beginning immediately — no preamble.
  2. On a NEW LINE, the literal token "Directions:" followed by exactly: no change
  3. On the FINAL LINE, the literal token "Reason:" followed by the reason kind(s) you
     actually used, comma-separated, from: baseline-already-reflects, residuals-contradict,
     reports-weak. Example:
       Directions: no change
       Reason: baseline-already-reflects, residuals-contradict
"""


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def row_class(r: dict) -> str:
    if r.get("operator") == "no-correction" or r.get("date_status") == "abstain":
        return "abstain"
    return "move" if r.get("operator") == "correction" else "no_change"


def dir_word(d: int) -> str:
    return {1: "increase", -1: "decrease", 0: "no change"}.get(d, "?")


def ensure_group2_prompt(prompt_dir: Path) -> str:
    """The group-2 prompt ships in code/data_augmentation/prompts/ (written from the default
    the first time if absent, editable there afterwards)."""
    d = AUG_PROMPT_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / GROUP2_PROMPT_NAME
    if not p.exists():
        p.write_text(GROUP2_PROMPT_DEFAULT)
        print(f"  preserved group-2 prompt → {p}")
    return p.read_text()


def strip_tail_lines(cot: str) -> str:
    """Drop the trailing 'Directions:' / 'Reason:' lines (finetune.py strips Directions/
    Sizes itself; the Reason line is bookkeeping only)."""
    lines = (cot or "").strip().splitlines()
    while lines and re.match(r"^\s*(Directions?|Sizes?|Reason)\s*:", lines[-1], re.I):
        lines.pop()
    return "\n".join(lines).strip()


def parse_reason_types(text: str) -> List[str]:
    m = _REASON_RE.search(text or "")
    if not m:
        return []
    toks = [t.strip().lower() for t in m.group(1).split(",")]
    return [t for t in toks if t in REASON_TYPES]


def sort_key_candidate(c: dict) -> Tuple:
    return (-(c.get("n_quotes_found") or 0), -(c.get("pro_grounding_score") or 0),
            int(c.get("round", 1)), int(c.get("candidate_index", 0)))


# ══════════════════════════════════════════════════════════════════════════════
# Classification
# ══════════════════════════════════════════════════════════════════════════════
def classify(row: dict, cands: List[dict]) -> Tuple[int, str, List[dict], List[dict]]:
    """-> (group, why, faithful_candidates, all_candidates_with_direction)."""
    truth = int((row.get("direction_labels") or [0])[0])
    with_dir = [c for c in cands if c.get("parsed_directions")]
    faithful = [c for c in with_dir if c.get("faithful_pass")]
    fdirs = {int(c["parsed_directions"][0]) for c in faithful}
    if faithful and fdirs == {0}:
        return 1, "faithful candidates all 'no change'" + (
            " (truth moved)" if truth != 0 else " (truth flat)"), faithful, with_dir
    if truth == 0:
        movers = [c for c in with_dir if int(c["parsed_directions"][0]) != 0]
        if movers:
            return 2, ("candidates argued a move (" + "/".join(sorted({dir_word(int(c['parsed_directions'][0])) for c in movers}))
                       + ") but the truth stayed within the band"), faithful, with_dir
        return 4, "truth flat but no candidate with a parsed direction", faithful, with_dir
    if faithful and len(fdirs) == 1 and 0 not in fdirs:
        d = next(iter(fdirs))
        return 3, (f"faithful candidates unanimous '{dir_word(d)}', truth "
                   f"'{dir_word(truth)}'" + (" (right direction, failed later)" if d == truth else " (wrong direction)")), faithful, with_dir
    if not faithful:
        return 4, "no faithful candidate; truth moved", faithful, with_dir
    return 4, "faithful candidates mixed move/no-change; truth moved", faithful, with_dir


# ══════════════════════════════════════════════════════════════════════════════
# Group 2: generation + judging
# ══════════════════════════════════════════════════════════════════════════════
def render_group2(row: dict, argument: dict, arg_text: str, tmpl: str,
                  lookback_len: int) -> str:
    return C.fill_template(tmpl, {
        "history_len": len(row["x_hist"]),
        "lookback_len": lookback_len,
        "nochange_band": C.fmt_nochange_band(MODE, BAND, DATA_MODE),
        "history_block": row["history_block"],
        "baseline_block": row["baseline_block"],
        "recent_residual_block": row["recent_residual_block"],
        "weekly_text_block": row["weekly_text_block"],
        "argument_direction": dir_word(int(argument["parsed_directions"][0])),
        "argument_block": strip_tail_lines(arg_text),
    })


def judge_candidate(llm, row: dict, cot: str, args) -> dict:
    """The pipeline's own gates, applied to a regenerated 'no change' reasoning. `llm` is
    the JUDGE client (thinking off, so the whole --judge-max-tokens budget is visible
    answer; with thinking on, Flash's reply was truncated before its 'Faithful:' line)."""
    res = {"parsed_directions": C.parse_directions(cot, HORIZON),
           "reason_types": parse_reason_types(cot),
           "template_reuse": TEMPLATE_MARK in cot}
    fdet = dquote_faithfulness(cot, row["weekly_text_block"])
    res.update({"n_quotes": fdet["n_quotes"], "n_quotes_found": fdet["n_found"],
                "missing_quotes": fdet["missing"], "verbatim_faithful": fdet["faithful"]})
    res["flash_faithful"] = None
    res["pro_grounding_score"] = None
    # the LLM judge decides faithfulness (semantic, as the pipeline does); the verbatim
    # check only gates candidates with no quote at all and serves as the no-API fallback
    ok_so_far = (res["parsed_directions"] == [0] and res["reason_types"]
                 and not res["template_reuse"] and fdet["n_quotes"] >= 1)
    if ok_so_far and llm is not None:
        subdir = C.prompt_mode(MODE, DATA_MODE)
        ftmpl = C.load_prompt("step1_faithfulness_flash.txt", args.prompt_dir, subdir)
        fprompt = C.fill_template(ftmpl, {"weekly_text_block": row["weekly_text_block"],
                                          "cot": strip_tail_lines(cot)})
        try:
            raw = llm.generate(model=args.flash_model, prompt=fprompt, temperature=0.0,
                               top_p=1.0, max_output_tokens=args.judge_max_tokens)
            res["flash_faithful"] = C.parse_faithful_verdict(raw)
            res["flash_truncated"] = res["flash_faithful"] is None
        except Exception as e:                       # noqa: BLE001
            logger.error("flash judge failed %s: %s", row["id"], e)
        if res["flash_faithful"]:
            gtmpl = C.load_prompt("step1_grounding_pro.txt", args.prompt_dir, subdir)
            gprompt = C.fill_template(gtmpl, {
                "weekly_text_block": row["weekly_text_block"],
                "recent_residual_block": row["recent_residual_block"],
                "directions": "no change", "horizon_len": HORIZON,
                "cot": strip_tail_lines(cot)})
            try:
                raw = llm.generate(model=args.pro_model, prompt=gprompt, temperature=0.0,
                                   top_p=1.0, max_output_tokens=args.judge_max_tokens)
                res["pro_grounding_score"] = C.parse_judge_score(raw)
                res["grounding_truncated"] = res["pro_grounding_score"] is None
            except Exception as e:                   # noqa: BLE001
                logger.error("grounding judge failed %s: %s", row["id"], e)
    res["faithful_pass"] = bool(ok_so_far and (res["flash_faithful"] if res["flash_faithful"] is not None
                                               else fdet["faithful"]))
    res["grounding_pass"] = (res["pro_grounding_score"] is not None
                             and res["pro_grounding_score"] >= args.min_grounding)
    res["accepted"] = bool(res["faithful_pass"] and res["grounding_pass"])
    res["fail_reason"] = (None if res["accepted"]
                          else "direction not 'no change'" if res["parsed_directions"] != [0]
                          else "no admissible Reason type" if not res["reason_types"]
                          else "reused the canned template" if res["template_reuse"]
                          else "no double-quoted phrase" if fdet["n_quotes"] < 1
                          else "flash faithfulness judge" if not res["faithful_pass"]
                          else f"grounding score {res['pro_grounding_score']} < {args.min_grounding}")
    return res


def run_group2(origins: List[Tuple[dict, dict, str]], llm, judge_llm, args,
               dst: Path) -> Dict[str, dict]:
    """origins: [(accepted_row, argument_candidate, argument_text)]. Generates --k
    candidates per origin (cache: group2_generated.jsonl), judges each (cache:
    group2_judged.jsonl), returns {id: best accepted candidate or None}."""
    tmpl = ensure_group2_prompt(args.prompt_dir)
    gen_path, jud_path = dst / "group2_generated.jsonl", dst / "group2_judged.jsonl"
    done_gen = {(r["id"], r["candidate_index"]): r for r in read_jsonl(gen_path)}
    done_jud = {(r["id"], r["candidate_index"]): r for r in read_jsonl(jud_path)}
    by_id = {row["id"]: (row, arg, txt) for row, arg, txt in origins}

    def n_cands(sid: str) -> int:
        return sum(1 for k in done_gen if k[0] == sid)

    def has_passer(sid: str) -> bool:
        return any(k[0] == sid and done_jud[k]["accepted"] for k in done_jud)

    def pending_ids() -> List[str]:
        """origins with no passer yet that may still get candidates (< --max-k)."""
        return [row["id"] for row, _, _ in origins
                if not has_passer(row["id"]) and n_cands(row["id"]) < args.max_k]

    def gen_one(item):
        sid, ci = item
        row, arg, txt = by_id[sid]
        prompt = render_group2(row, arg, txt, tmpl, args.lookback)
        try:
            cot = llm.generate(model=args.gen_model, prompt=prompt,
                               temperature=args.gen_temperature, top_p=C.GEN_TOP_P,
                               max_output_tokens=C.GEN_MAX_TOKENS)
        except Exception as e:                       # noqa: BLE001
            logger.error("gen failed %s ci=%d: %s", sid, ci, e)
            return None
        return {"id": sid, "candidate_index": ci, "state": row["state"],
                "reference_date": row["reference_date"], "cot": cot, "cot_chars": len(cot),
                "argument_candidate": {"candidate_index": arg.get("candidate_index"),
                                       "round": arg.get("round", 1),
                                       "direction": int(arg["parsed_directions"][0])}}

    def on_gen(r):
        if r is not None:
            append_jsonl(gen_path, r)
            done_gen[(r["id"], r["candidate_index"])] = r

    def judge_one(key):
        g = done_gen[key]
        row = by_id[key[0]][0]
        res = judge_candidate(judge_llm, row, g["cot"], args)
        return {"id": key[0], "candidate_index": key[1], "state": row["state"], **res}

    def on_jud(r):
        append_jsonl(jud_path, r)
        done_jud[(r["id"], r["candidate_index"])] = r

    # round 0 judges whatever is cached; each round generates --k more candidates ONLY for
    # origins that still have no passer and fewer than --max-k candidates
    rnd = 0
    while True:
        todo_j = [k for k in done_gen if k not in done_jud]
        if todo_j:
            print(f"  group 2 round {rnd}: judging {len(todo_j)} candidate(s) "
                  f"({len(done_jud)} cached)")
            C.run_parallel(todo_j, judge_one, args.workers, on_jud, label="group2 judge",
                           state_of=lambda r: r["state"])
        pend = pending_ids()
        if not pend:
            break
        todo = [(sid, ci) for sid in pend
                for ci in range(n_cands(sid), min(args.max_k, n_cands(sid) + args.k))]
        rnd += 1
        print(f"  group 2 round {rnd}: {len(pend)} origin(s) without a passer -> "
              f"generating {len(todo)} more candidate(s) (cap {args.max_k} per origin)")
        C.run_parallel(todo, gen_one, args.workers, on_gen, label="group2 gen",
                       state_of=lambda r: r["state"] if r else None)
        if not any(k not in done_jud for k in done_gen):
            break                                     # nothing new was generated (API errors)
    n_fail = sum(1 for row, _, _ in origins if not has_passer(row["id"]))
    print(f"  group 2: {len(origins) - n_fail}/{len(origins)} origins have a passing "
          f"candidate; {n_fail} dropped after up to {args.max_k} candidates")

    best: Dict[str, Optional[dict]] = {}
    for row, _, _ in origins:
        judged = sorted((done_jud[k] for k in done_jud if k[0] == row["id"]),
                        key=lambda j: j["candidate_index"])
        passers = [(j, done_gen[(j["id"], j["candidate_index"])]) for j in judged if j["accepted"]]
        passers.sort(key=lambda t: (-(t[0]["pro_grounding_score"] or 0),
                                    -(t[0]["n_quotes_found"] or 0), t[0]["candidate_index"]))
        best[row["id"]] = ({**passers[0][1], "judge": passers[0][0],
                            "n_tried": len(judged)} if passers else None)
        best[f"__attempts__{row['id']}"] = [{"candidate_index": j["candidate_index"],
                                             "accepted": j["accepted"],
                                             "fail_reason": j.get("fail_reason")} for j in judged]
    return best


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT)
    ap.add_argument("--dst", type=Path, default=None,
                    help="output dir (default <src>-abstainfix → --corpus-model gemini-abstainfix)")
    ap.add_argument("--prompt-dir", type=Path, default=PROMPT_DIR_DEFAULT)
    ap.add_argument("--k", type=int, default=3, help="group-2 candidates per origin per round")
    ap.add_argument("--max-k", type=int, default=6,
                    help="group-2: total candidates an origin may get before it is dropped "
                         "(extra rounds of --k only for origins with no passer yet)")
    ap.add_argument("--min-grounding", type=int, default=3)
    ap.add_argument("--gen-model", type=str, default=C.PRO_MODEL)
    ap.add_argument("--flash-model", type=str, default=C.FLASH_MODEL,
                    help="faithfulness judge (default: the pipeline's Flash model)")
    ap.add_argument("--pro-model", type=str, default=C.FLASH_MODEL,
                    help="grounding judge (default: Flash, as step1_filter.py does)")
    ap.add_argument("--gen-temperature", type=float, default=C.GEN_TEMPERATURE)
    ap.add_argument("--judge-max-tokens", type=int, default=4096,
                    help="output cap for the judge calls (their replies list every quote)")
    ap.add_argument("--lookback", type=int, default=C.RESID_LOOKBACK_DEFAULT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--thinking-budget", type=int, default=None)
    ap.add_argument("--thinking-level", type=str, default=None, choices=["low", "high"])
    ap.add_argument("--judge-group1", action="store_true",
                    help="also run the Flash faithfulness judge on each group-1 choice")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API calls: classify, build group 1, DROP group 2 (report only)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    C.setup_logging(args.verbose)

    src = args.src
    dst = args.dst or src.parent / f"{src.name}-abstainfix"
    if dst.resolve() == src.resolve():
        sys.exit("--dst must differ from --src (the source corpus is never modified)")
    dst.mkdir(parents=True, exist_ok=True)
    print(f"source : {src}\noutput : {dst}")

    accepted = list(read_jsonl(src / "accepted.jsonl"))
    if not accepted:
        sys.exit(f"no accepted.jsonl under {src}")
    cot_text: Dict[Tuple[str, int, int], str] = {}
    for fname in ("step1_generated.jsonl", "step1_generated_r2.jsonl"):
        for r in read_jsonl(src / fname):
            cot_text[(r["id"], int(r.get("candidate_index", -1)), int(r.get("round", 1)))] = r.get("cot", "")
    cands: Dict[str, List[dict]] = defaultdict(list)
    for r in read_jsonl(src / "step1_filtered_merged.jsonl"):
        cands[r["id"]].append(r)

    llm = judge_llm = None
    if not args.dry_run:
        llm = C.make_llm_client("gemini", thinking_budget=args.thinking_budget,
                                thinking_level=args.thinking_level)
        # judges: thinking OFF (Flash-tier) so the reply is not eaten by thinking tokens
        judge_llm = C.make_llm_client("gemini", thinking_budget=0)

    out_rows: List[dict] = []
    decisions: List[dict] = []
    stats: Counter = Counter()
    per_state: Dict[str, Counter] = defaultdict(Counter)
    group2_inputs: List[Tuple[dict, dict, str]] = []
    group2_rows: Dict[str, dict] = {}

    for r in accepted:
        cls = row_class(r)
        if cls != "abstain":
            out_rows.append(r)
            stats[cls] += 1
            per_state[r["state"]][cls] += 1
            continue
        group, why, faithful, with_dir = classify(r, cands.get(r["id"], []))
        dec = {"id": r["id"], "state": r["state"], "reference_date": r["reference_date"],
               "truth": int(r["direction_labels"][0]), "group": group, "why": why,
               "n_candidates": len(cands.get(r["id"], [])), "n_faithful": len(faithful)}
        stats[f"abstain_group{group}"] += 1
        per_state[r["state"]][f"group{group}"] += 1
        if group == 1:
            chosen = None
            for c in sorted(faithful, key=sort_key_candidate):
                text = cot_text.get((r["id"], int(c["candidate_index"]), int(c.get("round", 1))), "")
                if not text.strip():
                    continue
                if args.judge_group1 and judge_llm is not None:
                    subdir = C.prompt_mode(MODE, DATA_MODE)
                    ftmpl = C.load_prompt("step1_faithfulness_flash.txt", args.prompt_dir, subdir)
                    raw = judge_llm.generate(model=args.flash_model,
                                             prompt=C.fill_template(ftmpl, {"weekly_text_block": r["weekly_text_block"], "cot": strip_tail_lines(text)}),
                                             temperature=0.0, top_p=1.0, max_output_tokens=args.judge_max_tokens)
                    if C.parse_faithful_verdict(raw) is False:
                        dec.setdefault("judge_rejected", []).append(int(c["candidate_index"]))
                        continue
                chosen = (c, text)
                break
            if chosen is None:
                dec["decision"] = "dropped (no usable candidate text)"
                stats["abstain_group1_dropped"] += 1
                decisions.append(dec)
                continue
            c, text = chosen
            row = dict(r)
            row["abstain_cot"] = text
            row["step1_basis_cot"] = text
            row["basis_directions"] = [0]
            row["abstain_group"] = 1
            row["abstain_source"] = {"candidate_index": int(c["candidate_index"]),
                                     "round": int(c.get("round", 1)),
                                     "n_quotes_found": c.get("n_quotes_found")}
            out_rows.append(row)
            stats["abstain_kept_group1"] += 1
            per_state[r["state"]]["kept_group1"] += 1
            dec["decision"] = "kept (candidate reasoning)"
            dec["chosen"] = row["abstain_source"]
        elif group == 2:
            movers = [c for c in with_dir if int(c["parsed_directions"][0]) != 0]
            movers.sort(key=lambda c: (not c.get("faithful_pass"), *sort_key_candidate(c)))
            arg = None
            for c in movers:
                text = cot_text.get((r["id"], int(c["candidate_index"]), int(c.get("round", 1))), "")
                if text.strip():
                    arg = (c, text)
                    break
            if arg is None:
                dec["decision"] = "dropped (no argument text)"
                stats["abstain_group2_dropped"] += 1
                decisions.append(dec)
                continue
            dec["argument"] = {"candidate_index": int(arg[0]["candidate_index"]),
                               "round": int(arg[0].get("round", 1)),
                               "direction": dir_word(int(arg[0]["parsed_directions"][0])),
                               "faithful": bool(arg[0].get("faithful_pass"))}
            if args.dry_run:
                dec["decision"] = "dropped (--dry-run: regeneration skipped)"
                stats["abstain_group2_dropped"] += 1
            else:
                group2_inputs.append((r, arg[0], arg[1]))
                group2_rows[r["id"]] = r
                dec["decision"] = "pending regeneration"
        else:
            dec["decision"] = "dropped"
            stats[f"abstain_group{group}_dropped"] += 1
        decisions.append(dec)

    if group2_inputs:
        best = run_group2(group2_inputs, llm, judge_llm, args, dst)
        for dec in decisions:
            if dec.get("decision") != "pending regeneration":
                continue
            b = best.get(dec["id"])
            r = group2_rows[dec["id"]]
            dec["attempts"] = best.get(f"__attempts__{dec['id']}", [])
            if b is None:
                dec["decision"] = (f"dropped (none of {len(dec['attempts'])} regenerated "
                                   f"candidates passed the judges)")
                stats["abstain_group2_dropped"] += 1
                continue
            row = dict(r)
            row["abstain_cot"] = b["cot"]
            row["step1_basis_cot"] = b["cot"]
            row["basis_directions"] = [0]
            row["abstain_group"] = 2
            row["hard_negative"] = True
            row["abstain_reason_types"] = b["judge"]["reason_types"]
            row["abstain_source"] = {"regenerated_candidate_index": b["candidate_index"],
                                     "argument_candidate": b["argument_candidate"],
                                     "flash_faithful": b["judge"]["flash_faithful"],
                                     "pro_grounding_score": b["judge"]["pro_grounding_score"],
                                     "n_quotes_found": b["judge"]["n_quotes_found"]}
            out_rows.append(row)
            stats["abstain_kept_group2"] += 1
            per_state[r["state"]]["kept_group2"] += 1
            dec["decision"] = "kept (regenerated hard negative)"
            dec["chosen"] = row["abstain_source"]
            dec["reason_types"] = row["abstain_reason_types"]

    # deterministic order: state, date, id
    out_rows.sort(key=lambda x: (x.get("state", ""), x.get("reference_date", ""), x.get("id", "")))
    with (dst / "accepted.jsonl").open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    shutil.copy2(src / "bins.json", dst / "bins.json")
    with (dst / "abstain_fix_decisions.jsonl").open("w") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")

    n_ab = stats["abstain_kept_group1"] + stats["abstain_kept_group2"]
    total = stats["move"] + stats["no_change"] + n_ab
    # every target unique?
    texts = Counter((r.get("abstain_cot") or r.get("step1_basis_cot") or "") for r in out_rows)
    dup = sum(v - 1 for v in texts.values() if v > 1)
    report = {
        "source": str(src), "output": str(dst), "dry_run": args.dry_run,
        "rows": len(out_rows), "train_composition": {
            "move": stats["move"], "no_change": stats["no_change"],
            "abstain_group1_kept": stats["abstain_kept_group1"],
            "abstain_group2_kept": stats["abstain_kept_group2"],
            "abstain_share": (n_ab / total if total else None)},
        "abstain_groups": {f"group{g}": stats[f"abstain_group{g}"] for g in (1, 2, 3, 4)},
        "abstain_dropped": {f"group{g}": stats[f"abstain_group{g}_dropped"] for g in (1, 2, 3, 4)},
        "per_state": {s: dict(c) for s, c in sorted(per_state.items())},
        "duplicate_reasoning_texts": dup,
        "group2": {"k": args.k, "min_grounding": args.min_grounding, "gen_model": args.gen_model,
                   "flash_model": args.flash_model, "pro_model": args.pro_model},
    }
    (dst / "abstain_fix_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote  : {dst}/accepted.jsonl ({len(out_rows)} rows) + bins.json + "
          f"abstain_fix_decisions.jsonl + abstain_fix_report.json")
    print(f"abstain origins by group: 1={stats['abstain_group1']} 2={stats['abstain_group2']} "
          f"3={stats['abstain_group3']} 4={stats['abstain_group4']}")
    print(f"composition ({total} rows): move {stats['move']} ({stats['move']/total:.0%})  "
          f"no_change {stats['no_change']} ({stats['no_change']/total:.0%})  "
          f"abstain {n_ab} ({n_ab/total:.0%}) = group1 {stats['abstain_kept_group1']} + "
          f"group2 {stats['abstain_kept_group2']}  | dropped: group2 "
          f"{stats['abstain_group2_dropped']}, group3 {stats['abstain_group3_dropped']}, "
          f"group4 {stats['abstain_group4_dropped']}")
    print(f"duplicate reasoning texts among kept rows: {dup}")
    if args.dry_run:
        print("(--dry-run: group 2 was NOT regenerated; rerun without --dry-run for the API stage)")


if __name__ == "__main__":
    main()
