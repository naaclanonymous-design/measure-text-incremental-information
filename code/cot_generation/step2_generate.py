"""Step 2 — MAGNITUDE-BIN CoT generation (Semantic-Residual-Operator pipeline, 3/4).

For every date whose direction PASSED Step 1 and is a *move* (not no-change), this
re-shows the full five-block context PLUS the single carried Step-1 basis (its fixed
directions + reasoning) and samples k size-only CoTs from Gemini-3-pro. Each CoT picks
an UNSIGNED size class per horizon week (the direction is already fixed):
`small | medium | large | very-large | none`.

The one-directional |r| quantile bins (§2) are designed offline from the moving cases
in step1_generated.jsonl and written to bins.json (shared by Step-2 filtering).

Output (per state):
    output/CoT_generation_updated/<STATE>/step2_generated.jsonl
    output/CoT_generation_updated/<STATE>/bins.json

Run:
    python step2_generate.py --states GA,NJ,NY
"""
from __future__ import annotations

import argparse
import json
import sys
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


def resolve_bins(out_dir, args):
    """Resolve the size-class bins ONCE for the run. Default = GLOBAL bins pooled over
    EVERY state's residuals in this run's step1_filtered.jsonl (written to bins.json).
    --bins-json PATH uses a specific pre-built shared file instead."""
    names = [s.strip() for s in args.bin_names.split(",") if s.strip()]
    if args.bins_json:
        bins = json.loads(Path(args.bins_json).read_text())
        (out_dir / "bins.json").write_text(json.dumps(bins, indent=2))
        logger.info("using shared bins from %s", args.bins_json)
        return bins
    bins_split = None if args.bins_split == "" else args.bins_split
    return C.build_or_load_run_bins(
        out_dir, names=names, eps=args.nochange_band,
        bins_split=bins_split, rebuild=args.rebuild_bins,
        # influenza/finance chronos2_full: QUANTILE %-of-baseline bins;
        # covid_TSFM: QUANTILE |residual| bins in ABSOLUTE index points
        relative=C.band_is_relative(args.mode, args.data_mode),
        abs_index=(args.data_mode in C.COVID_DATA_MODES))


def render_step2(row, basis_dirs, basis_cot, bins, args) -> str:
    tmpl = C.load_prompt("step2_magnitude.txt", args.prompt_dir,
                         C.prompt_mode(args.mode, args.data_mode))
    return C.fill_template(tmpl, {
        "history_len": len(row["x_hist"]),
        "horizon_len": len(row["x_gt"]),
        "lookback_len": args.resid_lookback,
        "history_block": row["history_block"],
        "baseline_block": row["baseline_block"],
        "recent_residual_block": row["recent_residual_block"],
        # momentum-rescued dates carry the round-2 momentum summary; other dates render ''
        "momentum_block": C.momentum_section(row.get("momentum_text")),
        "weekly_text_block": row["weekly_text_block"],
        "step1_directions": directions_text(basis_dirs),
        "step1_reasoning": (basis_cot or "").strip(),
        "bin_definitions": C.fmt_bin_definitions(bins),
        "bin_names": C.bin_names_csv(bins),
    })


def size_row(item, gemini, bins, args):
    """One work unit = (move-date summary `m`, candidate_index). Renders the Step-2 prompt
    from `m` (which carries the 5 blocks + basis), calls Gemini, returns the row. Pure."""
    m, ci = item
    sid = m["id"]
    basis_dirs = m.get("basis_directions")
    prompt = render_step2(m, basis_dirs, m.get("basis_cot"), bins, args)
    if args.dry_run:
        cot = ""
    else:
        try:
            cot = gemini.generate(
                model=args.gen_model, prompt=prompt,
                temperature=args.gen_temperature, top_p=C.GEN_TOP_P,
                max_output_tokens=C.GEN_MAX_TOKENS)
        except Exception as e:
            logger.error("size-gen failed %s ci=%d: %s", sid, ci, e)
            return None   # write NO row — the candidate stays pending for a resume
    sizes = C.parse_sizes(cot, len(m["x_gt"]), bins)
    row = {
        "id": sid, "candidate_index": ci, "step": 2,
        "state": m.get("state"), "split": m.get("split"),
        "reference_date": m.get("reference_date"),
        "units": C.mode_units(args.mode, args.data_mode),
        "x_hat": m["x_hat"], "x_gt": m["x_gt"],
        **C.full_value_fields(m),   # *_full only in rate modes; count rows are complete already
        "basis_directions": basis_dirs,
        "weekly_text_block": m.get("weekly_text_block"),
        "recent_residual_block": m.get("recent_residual_block"),
        "cot": cot, "cot_chars": len(cot),
        "parsed_sizes": sizes,
    }
    if args.save_prompts or args.dry_run:
        row["prompt"] = prompt
    return row


def generate_run(gemini, out_dir, args, bins):
    """Reads ONLY step1_filtered.jsonl: the move-date SUMMARY rows already carry the full
    5-block context, the rate arrays, and the chosen Step-1 basis — so no
    step1_generated.jsonl is needed here. All (move-date × k) candidates across ALL states
    run concurrently."""
    flt_path = C.step1_filtered_view(out_dir)   # merged two-round view when the retry ran
    out_path = out_dir / "step2_generated.jsonl"
    if not flt_path.exists():
        logger.warning("missing %s — run step1_filter first.", flt_path)
        return
    logger.info("Step-1 source: %s", flt_path.name)

    moves = [r for r in C.iter_jsonl(flt_path)
             if int(r.get("candidate_index", 0)) == -1 and r.get("date_status") == "move"]
    done = C.load_done_pairs(out_path)
    pending = [(m, ci) for m in moves for ci in range(args.k)
               if (m["id"], ci) not in done]
    pend_by_state = {}
    for (m, _ci) in pending:
        st = m.get("state")
        pend_by_state[st] = pend_by_state.get(st, 0) + 1
    logger.info("STEP2-GEN: %d move-dates, %d candidates to size (k=%d) %s, %d-way parallel",
                len(moves), len(pending), args.k, pend_by_state, args.workers)

    writer = lambda row: C.append_jsonl(out_path, row) if row else None
    C.run_parallel(pending, lambda it: size_row(it, gemini, bins, args),
                   args.workers, writer, label="G2",
                   state_of=lambda row: row.get("state") if row else None)
    logger.info("Step-2 generation done (all states).")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-mode", type=str, default=C.DATA_MODE_DEFAULT,
                   choices=C.DATA_MODES,
                   help="input-data source (influenza | finance); selects the "
                        "matching output subdirectory")
    p.add_argument("--states", type=str, default=None)
    p.add_argument("--mode", type=str, default=C.MODE_DEFAULT, choices=C.MODES)
    p.add_argument("--k", type=int, default=C.K_DEFAULT)
    p.add_argument("--lookback", type=int, default=C.LOOKBACK_DEFAULT,
                   help="recent FluView report weeks (kept for parity with step1)")
    p.add_argument("--resid-lookback", type=int, default=C.RESID_LOOKBACK_DEFAULT,
                   help="recent-residual error window described in the prompt, default 3 weeks")
    p.add_argument("--nochange-band", type=float, default=None,
                   help="ε for bin design; default 0.01 per-10,000 (chronos2/flat), 0.10 "
                        "RELATIVE fraction of the baseline (chronos2_full), or 20 ABSOLUTE "
                        "index points (covid_TSFM)")
    p.add_argument("--bin-names", type=str, default=",".join(C.BIN_NAMES_DEFAULT))
    p.add_argument("--bins-split", type=str, default="train",
                   help="design bins on this split only (avoids test leakage); '' = all")
    p.add_argument("--bins-json", type=str, default=None,
                   help="use a specific pre-built shared bins.json instead of pooling")
    p.add_argument("--rebuild-bins", action="store_true",
                   help="re-pool and overwrite the cached run bins (bins.json)")
    p.add_argument("--model", type=str, default=C.PROVIDER_DEFAULT,
                   help="model family: 'gemini' (default, current setting) or 'GPT' "
                        "(most advanced GPT for pro-tier tasks, lighter GPT for flash-tier)")
    p.add_argument("--gen-model", type=str, default=None,
                   help="override the generation model (default: the provider's pro model)")
    p.add_argument("--gen-temperature", type=float, default=C.GEN_TEMPERATURE)
    p.add_argument("--workers", type=int, default=C.WORKERS_DEFAULT,
                   help="concurrent LLM calls across ALL states (1 = sequential)")
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
    p.add_argument("--save-prompts", action="store_true")
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
    args.gen_model = args.gen_model or C.provider_pro_model(args.provider)
    args.nochange_band = C.default_nochange_band(args.mode, args.nochange_band, args.data_mode)
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
    logger.info("run output dir (%s mode): %s", args.mode, out_dir)
    # Resolve the run's GLOBAL bins ONCE (pooled across all states in this run).
    bins = resolve_bins(out_dir, args)
    logger.info("bins scope=%s names=%s n_moving=%s",
                bins.get("scope"), bins["names"], bins.get("n_moving"))
    generate_run(gemini, out_dir, args, bins)
    logger.info("Run complete (Step-2 magnitude generation).")


if __name__ == "__main__":
    main()
