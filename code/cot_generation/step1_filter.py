"""Step 1 — DIRECTION CoT filtering (Semantic-Residual-Operator pipeline, stage 2/4).

Reads step1_generated.jsonl and, per origin date, applies the Step-1 gates from the
framework (§3 / §6):

  • OFFLINE direction correctness : parsed direction == sign(r) with the ±ε no-change
                                    band, for every horizon week.
  • OFFLINE self-consistency α₁   : agreement on the modal direction across the k CoTs;
                                    if α₁ < τ the date ABSTAINS (defers to the baseline).
  • LLM faithfulness  (EASY  → Gemini-2.5-Flash) : every quoted phrase appears verbatim
                                    in the scrubbed report text (a deterministic verbatim
                                    check is kept as a fallback / log).
  • LLM grounding     (COMPLEX → Gemini-3-pro)   : does the cited evidence genuinely
                                    support the stated direction per week (1–5 score)?

A passing date collapses its passers to ONE basis (the most-grounded / most-faithful
CoT) carried into Step 2. A correctly-called all-`no-change` date stops here (δ=0).

Output (per state):
    output/CoT_generation_updated/<STATE>/step1_filtered.jsonl
       - one row per candidate (verdicts), plus one date-summary row (candidate_index=-1)
         carrying date_status ∈ {move, no_change, abstain}, α₁, and the chosen basis.

Round 2 (momentum retry): --round 2 filters step1_generated_r2.jsonl (the regenerated
CoTs of the round-1 abstain dates, with the momentum summary in their prompt) into
step1_filtered_r2.jsonl, then writes step1_filtered_merged.jsonl: round-1 rows for the
dates that passed in round 1, round-2 rows for the retried dates. A date is only left
as ABSTAIN if it abstained in BOTH rounds. Step 2 reads the merged view automatically.

Run:
    python step1_filter.py --states GA,NJ,NY
    python step1_filter.py --states GA,NJ,NY --round 2     # refilter retries + merge
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]          # <paper-content>/code
for _d in ("cot_generation", "finetune", "zero_shot"):
    sys.path.insert(0, str(_CODE_ROOT / _d))
sys.path.insert(0, str(_CODE_ROOT))                         # paper_paths.py
import paper_paths as P                                     # noqa: E402
import cot_common as C
from cot_common import logger


def directions_text(dirs):
    name = {1: "increase", -1: "decrease", 0: "no change"}
    return ", ".join(name.get(d, "?") for d in (dirs or []))


def run_flash_faithfulness(gemini, row, args):
    """EASY task → Gemini-2.5-Flash: verbatim quote verification. Returns (verdict, raw)."""
    tmpl = C.load_prompt("step1_faithfulness_flash.txt", args.prompt_dir,
                         C.prompt_mode(args.mode, args.data_mode))
    prompt = C.fill_template(tmpl, {
        "weekly_text_block": row["weekly_text_block"], "cot": row["cot"].strip()})
    try:
        raw = gemini.generate(model=args.flash_model, prompt=prompt,
                              temperature=0.0, top_p=1.0,
                              max_output_tokens=C.JUDGE_MAX_TOKENS)
    except Exception as e:
        logger.error("flash faithfulness failed %s ci=%s: %s",
                     row["id"], row["candidate_index"], e)
        return None, ""
    return C.parse_faithful_verdict(raw), raw


def run_pro_grounding(gemini, row, args):
    """COMPLEX task → Gemini-3-pro: does the evidence support each week's direction?
    Returns (score 1–5 or None, raw)."""
    tmpl = C.load_prompt("step1_grounding_pro.txt", args.prompt_dir,
                         C.prompt_mode(args.mode, args.data_mode))
    prompt = C.fill_template(tmpl, {
        "weekly_text_block": row["weekly_text_block"],
        "recent_residual_block": row["recent_residual_block"],
        "directions": directions_text(row.get("parsed_directions")),
        "horizon_len": len(row["x_gt"]),
        "cot": row["cot"].strip()})
    try:
        raw = gemini.generate(model=args.pro_model, prompt=prompt,
                             temperature=0.0, top_p=1.0,
                             max_output_tokens=C.JUDGE_MAX_TOKENS)
    except Exception as e:
        logger.error("pro grounding failed %s ci=%s: %s",
                     row["id"], row["candidate_index"], e)
        return None, ""
    return C.parse_judge_score(raw), raw


# The input blocks are copied into each date-summary row so Step-2 generation can render
# purely from step1_filtered.jsonl (no step1_generated.jsonl needed downstream).
BLOCK_KEYS = ("history_block", "baseline_block", "recent_residual_block",
              "weekly_text_block")


def process_date(sid, rows, gemini, args):
    """Process ONE origin date: α₁, per-candidate gates (Flash faithfulness + Pro
    grounding), and basis selection. Returns (candidate_rows, summary_row). Pure — makes
    its own Gemini calls but writes no files, so it is safe to run many dates in parallel."""
    rows = sorted(rows, key=lambda r: r["candidate_index"])
    r0 = rows[0]
    state = r0["state"]
    H = len(r0["x_gt"])
    labels = r0["direction_labels"]
    dir_lists = [r["parsed_directions"] for r in rows if r.get("parsed_directions")]
    alpha1, modal = C.alpha1_direction(dir_lists)

    date_abstain = (not dir_lists) or (alpha1 < args.tau)
    cand_out = []
    passers = []
    if not date_abstain:
        for r in rows:
            pd_ = r.get("parsed_directions")
            n_correct = (sum(1 for a, b in zip(pd_, labels) if a == b) if pd_ else 0)
            need = min(args.min_correct_weeks, H) if args.min_correct_weeks else H
            dir_ok = bool(pd_) and n_correct >= need
            fdet = C.faithfulness(r["cot"], r["weekly_text_block"])
            flash_verdict, pro_score = None, None
            flash_raw, pro_raw = "", ""
            if dir_ok and not args.dry_run and gemini is not None:
                flash_verdict, flash_raw = run_flash_faithfulness(gemini, r, args)
                pro_score, pro_raw = run_pro_grounding(gemini, r, args)
            faithful_pass = (flash_verdict if flash_verdict is not None
                             else fdet["faithful"])
            ground_pass = (pro_score is None and not args.strict_grounding) or \
                          (pro_score is not None and pro_score >= args.min_grounding)
            accepted = dir_ok and faithful_pass and ground_pass
            cand = {
                "id": sid, "candidate_index": r["candidate_index"], "step": 1,
                "state": state, "split": r["split"],
                "reference_date": r["reference_date"],
                "parsed_directions": pd_, "direction_labels": labels,
                "n_correct_weeks": n_correct, "direction_ok": dir_ok,
                "verbatim_faithful": fdet["faithful"],
                "n_quotes": fdet["n_quotes"], "n_quotes_found": fdet["n_found"],
                "missing_quotes": fdet["missing"],
                "flash_faithful": flash_verdict, "pro_grounding_score": pro_score,
                "faithful_pass": faithful_pass, "grounding_pass": ground_pass,
                "accepted": accepted,
            }
            if accepted:
                cand["cot"] = r["cot"]          # record the FULL passing CoT
            if args.save_raw:
                cand["flash_raw"], cand["pro_raw"] = flash_raw, pro_raw
            cand_out.append(cand)
            if accepted:
                agree = sum(1 for a, m in zip(pd_, modal) if a == m)
                passers.append((r, cand, agree))

    if date_abstain or not passers:
        date_status = "abstain"
        basis_ci, basis_dirs, basis_cot = None, None, None
        passing_cots = []
    else:
        passers.sort(key=lambda t: (-t[1]["n_quotes_found"],
                                    -(t[1]["pro_grounding_score"] or 0), -t[2]))
        best_r, best_c, _ = passers[0]
        basis_ci = best_c["candidate_index"]
        basis_dirs = best_r["parsed_directions"]
        basis_cot = best_r["cot"]
        date_status = "no_change" if all(d == 0 for d in basis_dirs) else "move"
        passing_cots = [{
            "candidate_index": pr["candidate_index"], "cot": pr["cot"],
            "parsed_directions": pr["parsed_directions"],
            "n_quotes_found": pc["n_quotes_found"],
            "pro_grounding_score": pc["pro_grounding_score"],
            "is_basis": pr["candidate_index"] == basis_ci,
        } for (pr, pc, _ag) in passers]

    for c in cand_out:
        c["is_basis"] = (c["candidate_index"] == basis_ci)
        c["date_status"] = date_status
    summary = {
        "id": sid, "candidate_index": -1, "step": 1, "kind": "date_summary",
        "state": state, "split": r0["split"], "reference_date": r0["reference_date"],
        "n_fluview_reports": r0.get("n_fluview_reports"), "in_season": r0.get("in_season"),
        "units": r0.get("units", "per_10000"),
        # rate modes carry rate_decimals + *_full duplicates; chronos2_full rows keep the
        # complete full-precision values in x_* directly (no duplicate fields)
        **({} if r0.get("rate_decimals") is None
           else {"rate_decimals": r0["rate_decimals"]}),
        "x_hist": r0["x_hist"], "x_hat": r0["x_hat"], "x_gt": r0["x_gt"],
        **C.full_value_fields(r0),
        "alpha1": alpha1, "tau": args.tau, "modal_directions": modal,
        "direction_labels": labels, "date_status": date_status,
        "n_candidates": len(rows), "n_accepted": len(passers),
        "basis_candidate_index": basis_ci,
        "basis_directions": basis_dirs, "basis_cot": basis_cot,
        "passing_cots": passing_cots,
        "round": r0.get("round", 1),
    }
    if r0.get("momentum_text") is not None:
        summary["momentum_text"] = r0["momentum_text"]
    for k in BLOCK_KEYS:
        summary[k] = r0.get(k)
    return cand_out, summary


def filter_run(gemini, out_dir, args):
    sfx = C.round_suffix(args.round)
    gen_path = out_dir / f"step1_generated{sfx}.jsonl"
    flt_path = out_dir / f"step1_filtered{sfx}.jsonl"
    if not gen_path.exists():
        logger.warning("no %s — skipping.", gen_path)
        return

    by_id = defaultdict(list)
    for r in C.iter_jsonl(gen_path):
        by_id[r["id"]].append(r)

    done_ids = {r["id"] for r in C.iter_jsonl(flt_path) if int(r.get("candidate_index", 0)) == -1}
    todo = [(sid, by_id[sid]) for sid in by_id if sid not in done_ids]
    pend_by_state = {}
    for _sid, rows in todo:
        st = rows[0].get("state")
        pend_by_state[st] = pend_by_state.get(st, 0) + 1
    logger.info("STEP1-FILTER: %d dates done, %d to process %s, %d-way parallel",
                len(done_ids), len(todo), pend_by_state, args.workers)

    def on_done(res):
        cand_out, summary = res
        for c in cand_out:
            C.append_jsonl(flt_path, c)
        C.append_jsonl(flt_path, summary)

    C.run_parallel(todo, lambda it: process_date(it[0], it[1], gemini, args),
                   args.workers, on_done, label="F1",
                   state_of=lambda res: res[1].get("state"))

    summ = [r for r in C.iter_jsonl(flt_path) if int(r.get("candidate_index", 0)) == -1]
    counts = defaultdict(int)
    for r in summ:
        counts[r["date_status"]] += 1
    logger.info("Step-1 filter done (round %d, all states): %s", args.round, dict(counts))


def merge_rounds(out_dir):
    """Write step1_filtered_merged.jsonl — the two-round view Step 2 consumes:
      • dates that did NOT abstain in round 1 → their round-1 rows, unchanged;
      • round-1 abstain dates with a round-2 result → their round-2 rows (the summary
        keeps the round-2 date_status, so a passing retry is NOT labeled abstain);
      • round-1 abstain dates never retried → their round-1 rows (still abstain).
    Only dates that abstain in BOTH rounds remain abstain in the merged view."""
    p1 = out_dir / "step1_filtered.jsonl"
    p2 = out_dir / "step1_filtered_r2.jsonl"
    mp = out_dir / "step1_filtered_merged.jsonl"
    if not p1.exists():
        logger.warning("no %s — nothing to merge.", p1)
        return
    rows1, rows2 = defaultdict(list), defaultdict(list)
    status1, status2 = {}, {}
    for r in C.iter_jsonl(p1):
        rows1[r["id"]].append(r)
        if int(r.get("candidate_index", 0)) == -1:
            status1[r["id"]] = r.get("date_status")
    for r in C.iter_jsonl(p2):
        rows2[r["id"]].append(r)
        if int(r.get("candidate_index", 0)) == -1:
            status2[r["id"]] = r.get("date_status")

    counts = defaultdict(int)
    n_rescued = n_still_abstain = 0
    with mp.open("w") as f:
        for sid, r1rows in rows1.items():
            use_r2 = status1.get(sid) == "abstain" and sid in status2
            for r in (rows2[sid] if use_r2 else r1rows):
                out = dict(r)
                out.setdefault("round", 2 if use_r2 else 1)
                if use_r2 and int(r.get("candidate_index", 0)) == -1:
                    out["round1_status"] = "abstain"
                    out["rescued_by_momentum"] = status2[sid] != "abstain"
                f.write(json.dumps(out) + "\n")
            final = status2[sid] if use_r2 else status1.get(sid)
            counts[final] += 1
            if use_r2:
                if status2[sid] != "abstain":
                    n_rescued += 1
                else:
                    n_still_abstain += 1
    logger.info("MERGE → %s: final statuses %s (momentum rescued %d; abstained both rounds %d)",
                mp, dict(counts), n_rescued, n_still_abstain)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-mode", type=str, default=C.DATA_MODE_DEFAULT,
                   choices=C.DATA_MODES,
                   help="input-data source (influenza | finance); selects the "
                        "matching output subdirectory")
    p.add_argument("--states", type=str, default=None)
    p.add_argument("--mode", type=str, default=C.MODE_DEFAULT, choices=C.MODES)
    p.add_argument("--tau", type=float, default=C.TAU_DEFAULT,
                   help="abstain if α₁ < τ (default 0.6 ⇒ ≥3/5 CoTs must agree)")
    p.add_argument("--min-correct-weeks", type=int, default=2,
                   help="min horizon weeks whose direction must match the label (default 2, "
                        "capped at the horizon: chronos2 H=3 → 2-of-3 relaxation; flat H=2 → 2-of-2)")
    p.add_argument("--min-grounding", type=int, default=3,
                   help="minimum Gemini-3-pro grounding score (1–5) to accept")
    p.add_argument("--strict-grounding", action="store_true",
                   help="reject a candidate if the Pro grounding judge returns no score")
    p.add_argument("--round", type=int, default=1, choices=list(C.STEP1_ROUNDS),
                   help="1 = filter step1_generated.jsonl; 2 = filter the momentum-retry "
                        "candidates (step1_generated_r2.jsonl) and write the merged view")
    p.add_argument("--model", type=str, default=C.PROVIDER_DEFAULT,
                   help="model family: 'gemini' (default, current setting) or 'GPT' "
                        "(most advanced GPT for pro-tier tasks, lighter GPT for flash-tier)")
    p.add_argument("--pro-model", type=str, default=None,
                   help="override the grounding-judge model (default: the provider's "
                        "flash model, e.g. gemini-2.5-flash)")
    p.add_argument("--flash-model", type=str, default=None,
                   help="override the faithfulness-judge model (default: provider flash model)")
    p.add_argument("--workers", type=int, default=C.WORKERS_DEFAULT,
                   help="dates processed concurrently across ALL states (1 = sequential)")
    p.add_argument("--output-dir", type=Path, default=C.OUTPUT_DIR)
    p.add_argument("--scratch-root", type=Path, default=C.SCRATCH_OUTPUT_ROOT)
    p.add_argument("--prompt-dir", type=Path, default=None,
                   help="prompt set root (default: prompts/covid for covid modes, prompts/influenza otherwise)")
    p.add_argument("--batch", action="store_true",
                   help="send Gemini calls through the Vertex Batch API (~50%% token "
                        "price; needs GEMINI_BATCH_GCS_BUCKET; raises --workers to "
                        "cover all pending calls in one job)")
    p.add_argument("--thinking-budget", type=int, default=None,
                   help="Gemini thinking-token budget for ALL calls; 0 disables "
                        "thinking on 2.5-flash-tier models (2.5-pro floor: 128; "
                        "gemini-3* maps 0 to thinking-level low)")
    p.add_argument("--thinking-level", type=str, default=None, choices=["low", "high"],
                   help="gemini-3* thinking level (they cannot fully disable thinking)")
    p.add_argument("--credentials", type=Path, default=None,
                   help="API-key JSON (default: the provider's credentials file)")
    p.add_argument("--save-raw", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="offline gates only (no LLM faithfulness/grounding calls)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if getattr(args, "prompt_dir", None) is None:      # prompts/covid for covid modes, prompts/influenza otherwise
        args.prompt_dir = C.prompt_dir_for(getattr(args, "data_mode", "influenza"))
    return args


def main():
    args = parse_args()
    C.setup_logging(args.verbose)
    args.provider = C.normalize_provider(args.model)
    # grounding judge runs on the FLASH tier by default (cost saving; override
    # with --pro-model to restore the pro-tier judge)
    args.pro_model = args.pro_model or C.provider_flash_model(args.provider)
    args.flash_model = args.flash_model or C.provider_flash_model(args.provider)
    states = [s for s in (t.strip() for t in
                      (args.states or C.default_states(args.data_mode)).split(",")) if s]
    gemini = None if args.dry_run else C.make_llm_client(args.provider, args.credentials,
                                   thinking_budget=args.thinking_budget,
                                   thinking_level=args.thinking_level,
                                   use_batch=args.batch)
    if args.batch and args.workers < C.BATCH_WORKERS:
        logger.info("--batch: raising --workers %d -> %d so all pending calls "
                    "land in one batch job", args.workers, C.BATCH_WORKERS)
        args.workers = C.BATCH_WORKERS
    out_dir = C.scratch_run_dir(states, args.mode, args.output_dir, args.scratch_root,
                                provider=args.provider, data_mode=args.data_mode)
    logger.info("run output dir (%s mode, round %d, provider %s): %s",
                args.mode, args.round, args.provider, out_dir)
    filter_run(gemini, out_dir, args)
    if args.round > 1:
        merge_rounds(out_dir)
    logger.info("Run complete (Step-1 filtering, round %d).", args.round)


if __name__ == "__main__":
    main()
