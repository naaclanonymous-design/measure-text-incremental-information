"""Step 2 — MAGNITUDE-BIN CoT filtering + corpus finalize (pipeline stage 4/4).

For each move-date, every size CoT is grounded to a numerical correction with the
COLD-START lower-edge magnitude (no over-modification) and scored by UTILITY:

    δ[h] = sign(d[h]) · m_init(bin[h])             # d = Step-1 basis direction
    U    = 2⟨δ, r⟩ − ‖δ‖²                            # squared-error reduction (§6/§10)

Gates (per candidate):
  • OFFLINE utility            : U > margin  (direction is fixed → this checks the SIZE).
  • LLM faithfulness (EASY  → Gemini-2.5-Flash) : size-CoT quotes appear verbatim.
  • LLM size-grounding (COMPLEX → Gemini-3-pro) : does the evidence strength/speed match
                                  the chosen size class (1–5)?
The highest-U accepted candidate fixes the size operator; 0 accepted ⇒ abstain.

`finalize` then writes the full operator corpus accepted.jsonl, merging:
  move-dates (this script) + Step-1 no-change dates (δ=0) + abstain dates (no-correction).

Output (per state):
    output/CoT_generation_updated/<STATE>/step2_filtered.jsonl   (per-candidate verdicts)
    output/CoT_generation_updated/<STATE>/accepted.jsonl         (final operator corpus)

Run:
    python step2_filter.py --states GA,NJ,NY            # filter + finalize
    python step2_filter.py --states GA,NJ,NY --phase finalize
"""
from __future__ import annotations

import argparse
import json
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


def sizes_text(sizes, bins):
    out = []
    for s in (sizes or []):
        out.append("none" if s is None else bins["names"][s])
    return ", ".join(out)


def ground_delta(basis_dirs, sizes, bins, x_hat=None):
    """δ[h] = sign(d[h]) · m_init(bin[h]); 0 on no-change/none weeks.
    RELATIVE bins (chronos2_full): m_init is a FRACTION of the baseline, so
    δ[h] = sign(d[h]) · m_init(bin[h]) · x_hat[h] (in counts)."""
    rel = C.bins_are_relative(bins)
    delta = []
    for h, d in enumerate(basis_dirs or []):
        b = sizes[h] if (sizes and h < len(sizes)) else None
        if d == 0 or b is None:
            delta.append(0.0)
        else:
            mag = C.m_init(bins, b)
            if rel:
                mag *= abs(float(x_hat[h])) if (x_hat and h < len(x_hat)) else 0.0
            delta.append((1 if d > 0 else -1) * mag)
    return delta


def run_flash_faithfulness(gemini, row, args):
    tmpl = C.load_prompt("step2_faithfulness_flash.txt", args.prompt_dir,
                         C.prompt_mode(args.mode, args.data_mode))
    prompt = C.fill_template(tmpl, {
        "weekly_text_block": row["weekly_text_block"], "cot": row["cot"].strip()})
    try:
        raw = gemini.generate(model=args.flash_model, prompt=prompt, temperature=0.0,
                              top_p=1.0, max_output_tokens=C.JUDGE_MAX_TOKENS)
    except Exception as e:
        logger.error("flash faithfulness failed %s ci=%s: %s",
                     row["id"], row["candidate_index"], e)
        return None, ""
    return C.parse_faithful_verdict(raw), raw


def run_pro_grounding(gemini, row, sizes, bins, args):
    tmpl = C.load_prompt("step2_grounding_pro.txt", args.prompt_dir,
                         C.prompt_mode(args.mode, args.data_mode))
    prompt = C.fill_template(tmpl, {
        "weekly_text_block": row["weekly_text_block"],
        "recent_residual_block": row["recent_residual_block"],
        "directions": ", ".join({1: "increase", -1: "decrease", 0: "no change"}.get(d, "?")
                                for d in (row.get("basis_directions") or [])),
        "sizes": sizes_text(sizes, bins),
        "bin_definitions": C.fmt_bin_definitions(bins),
        "horizon_len": len(row["x_gt"]),
        "cot": row["cot"].strip()})
    try:
        raw = gemini.generate(model=args.pro_model, prompt=prompt, temperature=0.0,
                             top_p=1.0, max_output_tokens=C.JUDGE_MAX_TOKENS)
    except Exception as e:
        logger.error("pro size-grounding failed %s ci=%s: %s",
                     row["id"], row["candidate_index"], e)
        return None, ""
    return C.parse_judge_score(raw), raw


def process_move_date(sid, rows, gemini, bins, done, args):
    """Score every (undone) size candidate of ONE move-date: ground δ, utility U, then
    Flash faithfulness + Pro size-grounding. Returns the list of verdict rows. Pure."""
    rows = sorted(rows, key=lambda r: r["candidate_index"])
    r0 = rows[0]
    r = C.residuals(r0["x_hat"], r0["x_gt"])
    out_rows = []
    for rr in rows:
        ci = rr["candidate_index"]
        if (sid, ci) in done:
            continue
        sizes = rr.get("parsed_sizes")
        delta = ground_delta(rr.get("basis_directions"), sizes, bins,
                             x_hat=r0.get("x_hat"))
        U = C.utility(delta, r)
        u_pass = U > args.margin and any(d != 0 for d in delta)
        fdet = C.faithfulness(rr["cot"], rr["weekly_text_block"])
        flash_verdict, pro_score = None, None
        flash_raw, pro_raw = "", ""
        if u_pass and not args.dry_run and gemini is not None:
            flash_verdict, flash_raw = run_flash_faithfulness(gemini, rr, args)
            pro_score, pro_raw = run_pro_grounding(gemini, rr, sizes, bins, args)
        faithful_pass = flash_verdict if flash_verdict is not None else fdet["faithful"]
        ground_pass = (pro_score is None and not args.strict_grounding) or \
                      (pro_score is not None and pro_score >= args.min_grounding)
        accepted = u_pass and faithful_pass and ground_pass
        out = {
            "id": sid, "candidate_index": ci, "step": 2, "state": rr.get("state"),
            "split": rr["split"], "reference_date": rr["reference_date"],
            "basis_directions": rr.get("basis_directions"),
            "parsed_sizes": sizes, "residual": r,
            "delta": delta, "utility": U, "u_pass": u_pass,
            "verbatim_faithful": fdet["faithful"],
            "n_quotes": fdet["n_quotes"], "n_quotes_found": fdet["n_found"],
            "flash_faithful": flash_verdict, "pro_grounding_score": pro_score,
            "faithful_pass": faithful_pass, "grounding_pass": ground_pass,
            "accepted": accepted,
        }
        if args.save_raw:
            out["flash_raw"], out["pro_raw"] = flash_raw, pro_raw
        out_rows.append(out)
    return out_rows


def filter_run(gemini, out_dir, args):
    gen_path = out_dir / "step2_generated.jsonl"
    flt_path = out_dir / "step2_filtered.jsonl"
    bins_path = out_dir / "bins.json"
    if not gen_path.exists():
        logger.warning("no step2_generated.jsonl — skipping filter.")
        return
    if not bins_path.exists():
        logger.error("bins.json missing — run step2_generate first.")
        return
    bins = json.loads(bins_path.read_text())

    by_id = defaultdict(list)
    for r in C.iter_jsonl(gen_path):
        by_id[r["id"]].append(r)

    done = C.load_done_pairs(flt_path)
    todo = [(sid, rows) for sid, rows in by_id.items()
            if any((sid, rr["candidate_index"]) not in done for rr in rows)]
    pend_by_state = {}
    for _sid, rows in todo:
        st = rows[0].get("state")
        pend_by_state[st] = pend_by_state.get(st, 0) + 1
    logger.info("STEP2-FILTER: %d move-dates to score %s, %d-way parallel",
                len(todo), pend_by_state, args.workers)

    def on_done(out_rows):
        for o in out_rows:
            C.append_jsonl(flt_path, o)

    C.run_parallel(todo, lambda it: process_move_date(it[0], it[1], gemini, bins, done, args),
                   args.workers, on_done, label="F2",
                   state_of=lambda rows: rows[0].get("state") if rows else None)
    logger.info("Step-2 filtering done (all states).")


def finalize(out_dir, args):
    """Write the full operator corpus: move-dates (best-U accepted) + Step-1 no-change
    (δ=0) + abstain (no-correction, δ=0). Every row is SELF-CONTAINED for finetuning:
    it carries the four rendered model-input blocks, the full Step-1 basis CoT, the
    winning Step-2 size CoT (move dates), and the run's bin definitions — no join back
    to step1_filtered.jsonl / step2_generated.jsonl / bins.json is needed downstream."""
    flt_path = out_dir / "step2_filtered.jsonl"
    s1_path  = C.step1_filtered_view(out_dir)   # merged two-round view when the retry ran
    gen2_path = out_dir / "step2_generated.jsonl"
    acc_path = out_dir / "accepted.jsonl"
    bins_path = out_dir / "bins.json"
    bins = json.loads(bins_path.read_text()) if bins_path.exists() else None

    # best accepted size candidate per move-date
    by_id = defaultdict(list)
    for r in C.iter_jsonl(flt_path):
        by_id[r["id"]].append(r)
    best_move = {}
    for sid, rows in by_id.items():
        acc = [r for r in rows if r.get("accepted")]
        if acc:
            best_move[sid] = max(acc, key=lambda r: r["utility"])

    # winning Step-2 size CoT text, keyed (id, candidate_index)
    size_cots = {}
    if gen2_path.exists():
        for r in C.iter_jsonl(gen2_path):
            size_cots[(r["id"], int(r.get("candidate_index", -1)))] = r.get("cot")

    summaries = [r for r in C.iter_jsonl(s1_path)
                 if int(r.get("candidate_index", 0)) == -1]

    # the rendered model-INPUT blocks copied into every accepted row
    INPUT_BLOCK_KEYS = ("history_block", "baseline_block",
                        "recent_residual_block", "weekly_text_block")

    n_move = n_nochange = n_abstain = 0
    with acc_path.open("w") as acc:
        for s in summaries:
            sid = s["id"]
            base = {
                "id": sid, "state": s.get("state"), "split": s.get("split"),
                "reference_date": s.get("reference_date"),
                "n_fluview_reports": s.get("n_fluview_reports"),
                "in_season": s.get("in_season"),
                "units": s.get("units", "per_10000"),
                # rate modes: rate_decimals + *_full duplicates. chronos2_full: x_* ARE the
                # complete full-precision values — no duplicate fields written.
                **({} if s.get("rate_decimals") is None
                   else {"rate_decimals": s["rate_decimals"]}),
                "x_hist": s.get("x_hist"), "x_hat": s.get("x_hat"), "x_gt": s.get("x_gt"),
                **C.full_value_fields(s),
                "direction_labels": s.get("direction_labels"),
                "alpha1": s.get("alpha1"), "tau": s.get("tau"),
                # momentum-retry provenance: which Step-1 round produced this date's CoTs,
                # the momentum summary that was in the round-2 prompt (render the prompt
                # section from it with C.momentum_section when needed), whether the date
                # was rescued by the retry, and its round-1 status
                "step1_round": s.get("round", 1),
                **({} if s.get("momentum_text") is None
                   else {"momentum_text": s["momentum_text"]}),
                **({} if s.get("rescued_by_momentum") is None
                   else {"rescued_by_momentum": s["rescued_by_momentum"],
                         "round1_status": s.get("round1_status")}),
                # complete finetune INPUT: the exact blocks the CoTs were generated from
                **{k: s.get(k) for k in INPUT_BLOCK_KEYS},
                # run-level size-class definitions (also in bins.json; embedded for
                # self-containment)
                "bins": bins,
            }
            status = s.get("date_status")
            if status == "move" and sid in best_move:
                b = best_move[sid]
                sizes = b.get("parsed_sizes")
                acc.write(json.dumps({
                    **base, "operator": "correction", "date_status": "move",
                    "basis_directions": b.get("basis_directions"),
                    "sizes": sizes,
                    "size_names": sizes_text(sizes, bins) if bins else None,
                    "delta_cold_start": b.get("delta"),
                    "utility": b.get("utility"),
                    "residual": b.get("residual"),
                    "winning_candidate_index": b.get("candidate_index"),
                    "step1_basis_cot": s.get("basis_cot"),
                    # complete finetune TARGET reasoning: the winning size CoT text
                    "step2_size_cot": size_cots.get((sid, int(b.get("candidate_index", -1)))),
                }) + "\n")
                n_move += 1
            elif status == "no_change":
                acc.write(json.dumps({
                    **base, "operator": "no-change", "date_status": "no_change",
                    "basis_directions": s.get("basis_directions"),
                    "delta_cold_start": [0.0] * len(s.get("direction_labels") or []),
                    "cause": "the baseline is already accurate; keep it unchanged",
                    "step1_basis_cot": s.get("basis_cot"),
                }) + "\n")
                n_nochange += 1
            else:  # abstain, or a move-date with 0 accepted size CoTs
                acc.write(json.dumps({
                    **base, "operator": "no-correction", "date_status": "abstain",
                    "delta_cold_start": [0.0] * len(s.get("direction_labels") or []),
                    "cause": ("the text does not provide enough improvement over the "
                              "baseline; keep the baseline forecast unchanged"),
                }) + "\n")
                n_abstain += 1
    logger.info("FINALIZE → %s  (move=%d no_change=%d abstain=%d)",
                acc_path, n_move, n_nochange, n_abstain)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-mode", type=str, default=C.DATA_MODE_DEFAULT,
                   choices=C.DATA_MODES,
                   help="input-data source (influenza | finance); selects the "
                        "matching output subdirectory")
    p.add_argument("--states", type=str, default=None)
    p.add_argument("--mode", type=str, default=C.MODE_DEFAULT, choices=C.MODES)
    p.add_argument("--phase", choices=["all", "filter", "finalize"], default="all")
    p.add_argument("--margin", type=float, default=C.UTILITY_MARGIN,
                   help="accept a size CoT if U > margin (default 0)")
    p.add_argument("--min-grounding", type=int, default=3)
    p.add_argument("--strict-grounding", action="store_true")
    p.add_argument("--model", type=str, default=C.PROVIDER_DEFAULT,
                   help="model family: 'gemini' (default, current setting) or 'GPT' "
                        "(most advanced GPT for pro-tier tasks, lighter GPT for flash-tier)")
    p.add_argument("--pro-model", type=str, default=None,
                   help="override the size-grounding judge model (default: provider pro)")
    p.add_argument("--flash-model", type=str, default=None,
                   help="override the faithfulness judge model (default: provider flash)")
    p.add_argument("--workers", type=int, default=C.WORKERS_DEFAULT,
                   help="move-dates scored concurrently across ALL states (1 = sequential)")
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
    p.add_argument("--dry-run", action="store_true")
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
    need_llm = args.phase in ("all", "filter") and not args.dry_run
    gemini = C.make_llm_client(args.provider, args.credentials,
                                   thinking_budget=args.thinking_budget,
                                   thinking_level=args.thinking_level,
                                   use_batch=args.batch)
    if args.batch and args.workers < C.BATCH_WORKERS:
        logger.info("--batch: raising --workers %d -> %d so all pending calls "
                    "land in one batch job", args.workers, C.BATCH_WORKERS)
        args.workers = C.BATCH_WORKERS if need_llm else None
    out_dir = C.scratch_run_dir(states, args.mode, args.output_dir, args.scratch_root,
                                provider=args.provider, data_mode=args.data_mode)
    logger.info("run output dir (%s mode): %s", args.mode, out_dir)
    if args.phase in ("all", "filter"):
        filter_run(gemini, out_dir, args)
    if args.phase in ("all", "finalize"):
        finalize(out_dir, args)
    logger.info("Run complete (Step-2 filtering + finalize).")


if __name__ == "__main__":
    main()
