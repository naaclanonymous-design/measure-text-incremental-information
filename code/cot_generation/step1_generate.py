"""Step 1 — DIRECTION CoT generation (Semantic-Residual-Operator pipeline, stage 1/4).

For each leak-safe sample (one geography, one origin) this renders the input
blocks (12-week history, Chronos-2 baseline, 5-week recent residuals, raw
date-scrubbed FluView reports) and samples k direction-only CoTs from Gemini-3-pro.
Each CoT decides, per horizon week, only `increase | decrease | no change`, with the
±ε no-change band stated in the prompt. NO magnitude, NO ground truth in the input.

Output (per state, on /scratch, symlinked into /home):
    output/CoT_generation_updated/<STATE>/step1_generated.jsonl

Round 2 (momentum retry): --round 2 reads step1_filtered.jsonl, takes ONLY the dates
that ABSTAINED in round 1, writes an LLM ~50-word momentum summary of each date's
recent-residual block (momentum_summary.txt, cached in momentum_r2.jsonl), injects it
into the Step-1 prompt as item 5 ({momentum_block}), and regenerates k fresh CoTs into
step1_generated_r2.jsonl. step1_filter --round 2 then refilters and merges.

--model gemini (default) keeps the current Gemini setting; --model GPT switches every
call to OpenAI models (most advanced GPT for the pro-tier tasks, a lighter GPT for the
flash-tier tasks).

Run:
    python step1_generate.py --states GA,NJ,NY
    python step1_generate.py --states GA,NJ,NY --round 2      # momentum retry
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]          # <paper-content>/code
for _d in ("cot_generation", "finetune", "zero_shot"):
    sys.path.insert(0, str(_CODE_ROOT / _d))
sys.path.insert(0, str(_CODE_ROOT))                         # paper_paths.py
import paper_paths as P                                     # noqa: E402
import cot_common as C
from cot_common import logger


def render_step1(sample, blocks, args) -> str:
    tmpl = C.load_prompt("step1_direction.txt", args.prompt_dir,
                         C.prompt_mode(args.mode, args.data_mode))
    return C.fill_template(tmpl, {
        "history_len": len(sample["x_hist"]),
        "horizon_len": len(sample["x_gt"]),
        "lookback_len": args.resid_lookback,
        # rendered '0.01' for rate modes, '10%' for relative chronos2_full, '20 index
        # points' for covid_TSFM (absolute band)
        "nochange_band": C.fmt_nochange_band(args.mode, args.nochange_band, args.data_mode),
        "history_block": blocks["history_block"],
        "baseline_block": blocks["baseline_block"],
        "recent_residual_block": blocks["recent_residual_block"],
        # round 1: '' (prompt unchanged); round 2: the item-5 momentum summary section
        "momentum_block": blocks.get("momentum_block", ""),
        "weekly_text_block": blocks["weekly_text_block"],
    })


def ensure_momentum(id_to_inputs, llm, out_dir, args):
    """Round 2: one short LLM trend-summary per abstained date, rendered from the mode's
    momentum_summary.txt out of (a) the standardized directional-signal block and (b) the
    deterministic strength-weighted VERDICT block (see cot_common.assemble_momentum).
    Cached in momentum_r2.jsonl (resume-safe). id_to_inputs: {id: {"signal_block": str,
    "vote_block": str, "verdict": dict}}. Returns {id: momentum_text}."""
    path = out_dir / C.MOMENTUM_CACHE_NAME
    # ignore cached failures (empty text) so they are retried on resume
    have = {r["id"]: r["momentum"] for r in C.iter_jsonl(path) if r.get("momentum")}
    todo = [(sid, inp) for sid, inp in sorted(id_to_inputs.items()) if sid not in have]
    if not todo:
        return have
    tmpl = C.load_prompt(C.MOMENTUM_PROMPT_NAME, args.prompt_dir,
                         C.prompt_mode(args.mode, args.data_mode))
    logger.info("MOMENTUM: %d summaries to generate (model=%s)", len(todo), args.momentum_model)

    def work(item):
        sid, inp = item
        prompt = C.fill_template(tmpl, {"signal_block": inp["signal_block"],
                                        "vote_block": inp["vote_block"]})
        if args.dry_run:
            txt = ""
        else:
            try:
                txt = llm.generate(model=args.momentum_model, prompt=prompt,
                                   temperature=C.MOMENTUM_TEMPERATURE, top_p=1.0,
                                   max_output_tokens=C.MOMENTUM_MAX_TOKENS)
            except Exception as e:
                logger.error("momentum failed %s: %s", sid, e)
                return None   # cache NOTHING — the summary is retried on resume
        # provenance: the exact signals + verdict the paragraph was rendered from
        return {"id": sid, "momentum": txt.strip(), "model": args.momentum_model,
                "signal_block": inp["signal_block"], "vote_block": inp["vote_block"],
                "verdict": inp["verdict"]}

    def on_done(row):
        if row is None:
            return
        C.append_jsonl(path, row)
        have[row["id"]] = row["momentum"]

    C.run_parallel(todo, work, args.workers, on_done, label="M")
    return have


def load_abstain_ids(out_dir):
    """Round-1 abstained dates (date_status == 'abstain' in step1_filtered.jsonl)."""
    flt = out_dir / "step1_filtered.jsonl"
    if not flt.exists():
        sys.exit(f"--round 2 needs the round-1 filter output first: {flt} is missing")
    return {r["id"] for r in C.iter_jsonl(flt)
            if int(r.get("candidate_index", 0)) == -1 and r.get("date_status") == "abstain"}


def build_blocks(sample, cdc_store, cache, scale, args) -> dict:
    ref = sample["reference_date"]
    hist_dates = C.history_dates_for(ref, len(sample["x_hist"]))
    lb_dates   = C.lookback_dates_for(ref, args.lookback)
    dec = C.mode_value_decimals(args.mode, args.data_mode)   # 2dp rates / whole counts / cents
    # relative chronos2_full (influenza/finance): residuals as CHANGING RATIOS (%);
    # covid_TSFM: absolute index-point residuals (band_is_relative False)
    rel = C.band_is_relative(args.mode, args.data_mode)
    # TWO recent-residual horizons: the 1-week-ahead (next-week) and 3-week-ahead Chronos-2
    # forecast errors over the last `lookback` observed weeks. covid_TSFM_updated is
    # h+3-only end to end: it keeps ONLY the far-horizon (3-week-ahead) errors.
    far_only = args.data_mode == "covid_TSFM_updated"
    recent1 = (None if far_only else
               C.build_recent_residuals(sample["state"], ref, sample["x_hist"],
                                        cache, scale, args.resid_lookback, step_h=1,
                                        decimals=dec, relative=rel))
    recent3 = C.build_recent_residuals(sample["state"], ref, sample["x_hist"],
                                       cache, scale, args.resid_lookback, step_h=3,
                                       decimals=dec, relative=rel)
    desc1 = ("(actual − the 1-step forecast) as a PERCENTAGE of that forecast "
             "(+ = forecast too low)" if rel
             else "actual minus the 1-step forecast (+ = forecast too low)")
    desc3 = ("(actual − the 3-step forecast) as a PERCENTAGE of that forecast "
             "(+ = forecast too low)" if rel
             else "actual minus the 3-step forecast (+ = forecast too low)")
    if far_only:
        combined_recent = (
            f"Chronos-2 3-WEEK-AHEAD recent errors — {desc3}:\n"
            + C.fmt_recent_residual_block(recent3, decimals=dec, relative=rel))
    else:
        combined_recent = (
            f"Chronos-2 NEXT-WEEK (1-week-ahead) recent errors — {desc1}:\n"
            + C.fmt_recent_residual_block(recent1, decimals=dec, relative=rel)
            + f"\n\nChronos-2 3-WEEK-AHEAD recent errors — {desc3}:\n"
            + C.fmt_recent_residual_block(recent3, decimals=dec, relative=rel))
    report_entries = cdc_store.for_dates(lb_dates)
    return {
        "history_block": C.fmt_rate_block(hist_dates, sample["x_hist"], decimals=dec),
        "baseline_block": C.fmt_baseline_block(
            sample["x_hat"], decimals=dec, offsets=sample.get("pred_offsets"),
            label_by_offset=(args.data_mode == "finance_TSFM"
                             or args.data_mode in C.COVID_DATA_MODES)),
        "recent_residual_block": combined_recent,
        "weekly_text_block": C.fmt_weekly_text_block(report_entries, ref,
                                                     C.report_kind(args.data_mode)),
        "_recent_residuals": {"next_week": recent1, "three_week": recent3},
        # round-2 SIGNAL C source — influenza: ILINet %-ILI trend from the FluView
        # reports; finance modes: net bullish/bearish market tone of the news reports
        "_surveillance": (C.extract_market_sentiment(report_entries)
                          if args.data_mode in C.FINANCE_DATA_MODES
                          else C.extract_surveillance(report_entries)),
    }


def gen_row(item, gemini, args):
    """One work unit = (state, sample, prebuilt blocks, candidate_index). Renders the
    Step-1 prompt, calls Gemini, returns the row. Pure (no file writes) → thread-safe."""
    state, sample, blocks, ci = item
    prompt = render_step1(sample, blocks, args)
    if args.dry_run:
        cot = ""
    else:
        try:
            cot = gemini.generate(
                model=args.gen_model, prompt=prompt,
                temperature=args.gen_temperature, top_p=C.GEN_TOP_P,
                max_output_tokens=C.GEN_MAX_TOKENS)
        except Exception as e:
            logger.error("gen failed %s ci=%d: %s", sample["id"], ci, e)
            return None   # write NO row — the candidate stays pending for a resume
    parsed = C.parse_directions(cot, len(sample["x_gt"]))
    row = {
        "id": sample["id"], "candidate_index": ci, "step": 1,
        "split": sample["split"], "state": state,
        "reference_date": sample["reference_date"],
        "n_fluview_reports": sample.get("n_fluview_reports"),
        "in_season": sample.get("in_season"),
        "units": C.mode_units(args.mode, args.data_mode),
        # rate modes: 2dp working values + *_full originals. chronos2_full: x_* ARE the
        # complete full-precision values — no rate_decimals, no *_full duplicates.
        **({} if C.mode_is_count(args.mode)
           else {"rate_decimals": C.RATE_ROUND_DECIMALS}),
        "x_hist": sample["x_hist"], "x_hat": sample["x_hat"], "x_gt": sample["x_gt"],
        **C.full_value_fields(sample),
        # absolute rate band (chronos2/flat) or RELATIVE ±10% band (chronos2_full)
        "direction_labels": C.direction_labels_for_mode(args.mode, sample["x_hat"],
                                                        sample["x_gt"], args.nochange_band,
                                                        args.data_mode),
        "recent_residuals": blocks["_recent_residuals"],
        "history_block": blocks["history_block"],
        "baseline_block": blocks["baseline_block"],
        "recent_residual_block": blocks["recent_residual_block"],
        "weekly_text_block": blocks["weekly_text_block"],
        "cot": cot, "cot_chars": len(cot),
        "parsed_directions": parsed,
        "round": args.round,
    }
    if args.round > 1:
        row["momentum_text"] = blocks.get("_momentum_text", "")
    if args.save_prompts or args.dry_run:
        row["prompt"] = prompt
    return row


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-mode", type=str, default=C.DATA_MODE_DEFAULT, choices=C.DATA_MODES,
                   help="input-data source: 'influenza' (per-state CSVs in data/Influenza) "
                        "or 'finance_stat' (per-ticker CSVs in weekly_with_news/organized, "
                        "stat_pred_3w baseline). The generation method is identical; only the "
                        "input data differs. finance_stat requires --mode chronos2_full")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="override the data-mode's input CSV directory")
    p.add_argument("--states", type=str, default=None,
                   help="geo codes (influenza, default GA,NJ,NY) or tickers (finance, "
                        "default AAPL,GOOGL,NVDA)")
    p.add_argument("--mode", type=str, default=C.MODE_DEFAULT, choices=C.MODES,
                   help="chronos2 = 3-week Chronos-2 horizon (per-10,000 rates); flat = "
                        "predict ref+1 & ref+3 off a flat baseline; chronos2_full = chronos2 "
                        "on RAW hospitalization counts with a RELATIVE ±10%% no-change band "
                        "(covid_TSFM: ABSOLUTE ±20-index-point band)")
    p.add_argument("--splits", type=str, default=C.SPLITS_DEFAULT)
    p.add_argument("--k", type=int, default=C.K_DEFAULT)
    p.add_argument("--lookback", type=int, default=C.LOOKBACK_DEFAULT,
                   help="how many recent FluView report weeks to show + gate on")
    p.add_argument("--resid-lookback", type=int, default=C.RESID_LOOKBACK_DEFAULT,
                   help="recent-residual error window (Chronos-2 / GT-change), default 3 weeks")
    p.add_argument("--nochange-band", type=float, default=None,
                   help="no-change band ε; default 0.01 per-10,000 (chronos2/flat), 0.10 "
                        "RELATIVE fraction of the baseline (chronos2_full), or 20 ABSOLUTE "
                        "index points (covid_TSFM)")
    p.add_argument("--min-fluview-weeks", type=int, default=1)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--sample-date-from", type=str, default="none")
    p.add_argument("--sample-date-to", type=str, default="none")

    p.add_argument("--round", type=int, default=1, choices=list(C.STEP1_ROUNDS),
                   help="1 = normal generation; 2 = momentum retry of the dates that "
                        "ABSTAINED in the round-1 filter (writes *_r2 files)")
    p.add_argument("--model", type=str, default=C.PROVIDER_DEFAULT,
                   help="model family: 'gemini' (default, current setting) or 'GPT' "
                        "(most advanced GPT for pro-tier tasks, lighter GPT for flash-tier)")
    p.add_argument("--gen-model", type=str, default=None,
                   help="override the generation model (default: the provider's pro model)")
    p.add_argument("--momentum-model", type=str, default=None,
                   help="override the round-2 momentum-summary model (default: the "
                        "provider's pro model — Gemini-3.1-pro-preview under gemini)")
    p.add_argument("--gen-temperature", type=float, default=C.GEN_TEMPERATURE)
    p.add_argument("--workers", type=int, default=C.WORKERS_DEFAULT,
                   help="concurrent LLM calls across ALL states (1 = sequential)")

    p.add_argument("--chronos2-cache", type=Path, default=C.CACHE_PATH_STATES)
    p.add_argument("--usa-cache", type=Path, default=C.CACHE_PATH_USA)
    p.add_argument("--cdc-dir", type=Path, default=C.CDC_DIR)
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
    p.add_argument("--target-hosp-csv", type=Path, default=C.TARGET_HOSP_CSV)
    p.add_argument("--output-dir", type=Path, default=C.OUTPUT_DIR)
    p.add_argument("--scratch-root", type=Path, default=C.SCRATCH_OUTPUT_ROOT)
    p.add_argument("--prompt-dir", type=Path, default=None,
                   help="prompt set root (default: prompts/covid for covid modes, prompts/influenza otherwise)")

    p.add_argument("--save-prompts", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if getattr(args, "prompt_dir", None) is None:      # prompts/covid for covid modes, prompts/influenza otherwise
        args.prompt_dir = C.prompt_dir_for(getattr(args, "data_mode", "influenza"))
    return args


def _prep_state_samples(state, splits, builder, pop_map, cdc_store, args, date_from, date_to):
    """Build + gate + date-filter + limit + unit-convert + round samples for one state.
    `builder` is build_chronos2_samples or build_flat_samples. Returns (samples, cache, scale).
    chronos2/flat: counts → per-10,000 rates, 2dp. chronos2_full: NO transform — whole-number
    counts (scale=1.0 so the Chronos-2 cache residuals stay in counts too)."""
    requested = {s.strip().lower() for s in splits}
    if args.bundle.split_ranges is None:
        # influenza: 'val' is carved out of the train date range, so requesting val
        # means building the train range; the final-split filter below then keeps
        # exactly the requested splits (e.g. default 'train' excludes val/test).
        build_splits = sorted({"train" if s == "val" else s for s in requested})
    else:
        build_splits = sorted(requested)
    samples, cache = builder([args.chronos2_cache, args.usa_cache], [state], build_splits,
                             bundle=args.bundle)
    samples = [s for s in samples if s["split"] in requested]
    if args.data_mode == "finance_TSFM":
        C.subset_samples_to_t3(samples)          # keep ONLY the origin+3 horizon
    elif args.data_mode in C.COVID_DATA_MODES:
        C.subset_samples_to_offsets(samples, C.covid_pred_offsets(args.data_mode))
    samples = C.tag_and_gate_fluview(samples, cdc_store, args.lookback, args.min_fluview_weeks)
    if date_from or date_to:
        samples = [s for s in samples
                   if (date_from is None or s["reference_date"] >= date_from)
                   and (date_to is None or s["reference_date"] <= date_to)]
    samples.sort(key=lambda s: s["reference_date"], reverse=True)
    if args.limit is not None and args.limit >= 0:
        samples = samples[:args.limit]
    if C.mode_is_count(args.mode):
        # actual counts stay as-is (NHSN whole numbers); the Chronos-2 BASELINE is rounded
        # to whole admissions (matching the display + integer recent residuals). No *_full
        # duplicates — these ARE the complete working values.
        scale = 1.0
        C.round_baseline_counts(samples, C.mode_value_decimals(args.mode, args.data_mode))
    else:
        scale = C.RATE_PER / pop_map[state.upper()]
        C.to_rates(samples, pop_map[state.upper()])
        C.round_sample_rates(samples)    # 2dp working values + *_full copies
    return samples, cache, scale


def run_chronos2(states, splits, pop_map, gemini, out_dir, args, date_from, date_to,
                 abstain_ids=None):
    gen_path = out_dir / f"step1_generated{C.round_suffix(args.round)}.jsonl"
    done = C.load_done_pairs(gen_path)
    prepared = []
    for state in states:
        cdc_store = args.bundle.store_for(state)
        samples, cache, scale = _prep_state_samples(
            state, splits, C.build_chronos2_samples, pop_map, cdc_store, args, date_from, date_to)
        if abstain_ids is not None:
            samples = [s for s in samples if s["id"] in abstain_ids]
        if not samples:
            logger.warning("[%s] no samples after windowing — skipping.", state)
            continue
        for s in samples:
            prepared.append((state, s, build_blocks(s, cdc_store, cache, scale, args)))
        logger.info("[%s] samples=%d", state, len(samples))
    if args.round > 1:
        # one directional-signal + strength-weighted-verdict summary per abstained date,
        # injected as prompt item 5. Signals + vote are built in cot_common.assemble_momentum
        # from GT history (analogs, level, acceleration, neighbors), the recent residuals, and
        # the parsed surveillance trend.
        hists = {st: args.bundle.histories[st]
                 for st in sorted({s for s, _x, _b in prepared})
                 if st in args.bundle.histories}
        inputs = {}
        for st, s, blocks in prepared:
            eps = (C.build_analog_episodes(hists[st], s["reference_date"])
                   if st in hists else [])
            rr1 = blocks["_recent_residuals"]["next_week"]
            rr3 = blocks["_recent_residuals"]["three_week"]
            # covid_TSFM_updated (rr1 is None): far-horizon errors drive Signal A
            # directly, with no near-horizon corroborator.
            mom = C.assemble_momentum(
                state=st, reference_date=s["reference_date"],
                recent1=(rr1 if rr1 is not None else rr3),
                recent3=(rr3 if rr1 is not None else None), is_flat=False,
                episodes=eps, surveillance=blocks.get("_surveillance"),
                hist=hists.get(st), neighbor_hists=hists, lookback=args.lookback)
            if mom is not None:
                inputs[s["id"]] = mom
        momentum = ensure_momentum(inputs, gemini, out_dir, args)
        for _st, s, blocks in prepared:
            blocks["_momentum_text"] = momentum.get(s["id"], "")
            blocks["momentum_block"] = C.momentum_section(blocks["_momentum_text"])
    pending = [(state, s, blocks, ci)
               for state, s, blocks in prepared for ci in range(args.k)
               if (s["id"], ci) not in done]
    logger.info("STEP1-GEN (chronos2, round %d): %d candidates, %d-way parallel",
                args.round, len(pending), args.workers)
    writer = lambda row: C.append_jsonl(gen_path, row) if row else None
    C.run_parallel(pending, lambda it: gen_row(it, gemini, args),
                   args.workers, writer, label="G",
                   state_of=lambda row: row.get("state") if row else None)


# ─────────────────────────── FLAT mode ───────────────────────────
def build_flat_packs(s, cdc_store, cache, scale, args):
    """Shared blocks + per-predicted-week (flat baseline, GT-change recent residuals) for a
    flat sample, plus the COMBINED 2-week baseline/recent blocks used by the combined row.
    Residuals are the GT change over the matching horizon (step=1 for ref+1, step=3 for
    ref+3); the cache/scale are unused in flat mode."""
    ref = s["reference_date"]
    hist_dates = C.history_dates_for(ref, len(s["x_hist"]))
    lb_dates   = C.lookback_dates_for(ref, args.lookback)
    perweek = []
    for w_idx, (off, step) in enumerate(zip(C.FLAT_PRED_OFFSETS, C.FLAT_RESID_STEPS)):
        rr = C.build_recent_residuals_diff(ref, s["x_hist"], args.resid_lookback, step)
        perweek.append({
            "offset": off, "step": step, "recent": rr,
            "baseline_block": C.fmt_baseline_block([s["x_hat"][w_idx]], offsets=[off]),
            "recent_block": C.fmt_recent_change_block(rr, step),
        })
    combined_recent = (
        f"Recent {C.FLAT_RESID_STEPS[0]}-week changes in the admission rate "
        f"(matching the 1-week-ahead target, predicted week origin+{C.FLAT_PRED_OFFSETS[0]}):\n"
        + perweek[0]["recent_block"]
        + f"\n\nRecent {C.FLAT_RESID_STEPS[1]}-week changes in the admission rate "
        f"(matching the 3-week-ahead target, predicted week origin+{C.FLAT_PRED_OFFSETS[1]}):\n"
        + perweek[1]["recent_block"])
    # NOTE: round 1 has no momentum block (redundant with the recent GT-change residuals);
    # round 2 injects an LLM summary of the COMBINED two-horizon changes (s["_momentum"]).
    report_entries = cdc_store.for_dates(lb_dates)
    return {
        "history_block": C.fmt_rate_block(hist_dates, s["x_hist"]),
        "weekly_text_block": C.fmt_weekly_text_block(report_entries, ref,
                                                     C.report_kind(args.data_mode)),
        "combined_baseline": C.fmt_baseline_block(s["x_hat"], offsets=list(C.FLAT_PRED_OFFSETS)),
        "combined_recent": combined_recent,
        "perweek": perweek,
        # round-2 SIGNAL C source: ILINet %-ILI trend parsed from the FluView reports
        "_surveillance": C.extract_surveillance(report_entries),
    }


def flat_sw_row(item, gemini, args):
    """Generate ONE single-week direction CoT for predicted week `w_idx` of a flat sample."""
    s, w_idx, ci = item
    packs = s["_packs"]; pw = packs["perweek"][w_idx]
    pseudo = {"x_hist": s["x_hist"], "x_gt": [s["x_gt"][w_idx]]}    # horizon_len = 1
    blocks = {"history_block": packs["history_block"], "baseline_block": pw["baseline_block"],
              "recent_residual_block": pw["recent_block"],
              "weekly_text_block": packs["weekly_text_block"],
              # round 2: the date-level momentum summary (covers BOTH horizon spans)
              "momentum_block": C.momentum_section(s.get("_momentum"))}
    prompt = render_step1(pseudo, blocks, args)
    if args.dry_run:
        cot = ""
    else:
        try:
            cot = gemini.generate(model=args.gen_model, prompt=prompt,
                                  temperature=args.gen_temperature, top_p=C.GEN_TOP_P,
                                  max_output_tokens=C.GEN_MAX_TOKENS)
        except Exception as e:
            logger.error("flat gen failed %s w%d ci=%d: %s", s["id"], w_idx, ci, e)
            return None   # write NO row — the single-week half stays pending for a resume
    d = C.parse_directions(cot, 1)
    row = {"id": s["id"], "candidate_index": ci, "week_index": w_idx, "state": s["state"],
           "split": s["split"], "reference_date": s["reference_date"],
           "offset": pw["offset"], "resid_step": pw["step"],
           "cot": cot, "cot_chars": len(cot),
           "parsed_direction": (d[0] if d else None)}
    if args.save_prompts or args.dry_run:
        row["prompt"] = prompt
    return row


def combine_flat(sw_path, gen_path, sample_by_id, args):
    """Pair the two single-week runs candidate-wise into one combined 2-week direction row."""
    from collections import defaultdict
    gen_done = C.load_done_pairs(gen_path)
    by_id = defaultdict(dict)
    for r in C.iter_jsonl(sw_path):
        by_id[r["id"]][(int(r["week_index"]), int(r["candidate_index"]))] = r
    n = 0
    for sid, cells in by_id.items():
        s = sample_by_id.get(sid)
        if s is None:
            continue
        packs = s["_packs"]
        for ci in range(args.k):
            if (sid, ci) in gen_done:
                continue
            w0, w1 = cells.get((0, ci)), cells.get((1, ci))
            if not (w0 and w1):
                continue                      # both single-week halves must exist
            d0, d1 = w0.get("parsed_direction"), w1.get("parsed_direction")
            parsed = [d0, d1] if (d0 is not None and d1 is not None) else None
            cot = (f"=== Predicted week origin+{C.FLAT_PRED_OFFSETS[0]} (1-week-ahead) ===\n"
                   + (w0.get("cot") or "")
                   + f"\n\n=== Predicted week origin+{C.FLAT_PRED_OFFSETS[1]} (3-week-ahead) ===\n"
                   + (w1.get("cot") or ""))
            row = {
                "id": sid, "candidate_index": ci, "step": 1, "mode": "flat",
                "split": s["split"], "state": s["state"], "reference_date": s["reference_date"],
                "n_fluview_reports": s.get("n_fluview_reports"), "in_season": s.get("in_season"),
                "units": "per_10000", "rate_decimals": C.RATE_ROUND_DECIMALS,
                "pred_offsets": list(C.FLAT_PRED_OFFSETS),
                "x_hist": s["x_hist"], "x_hat": s["x_hat"], "x_gt": s["x_gt"],
                "x_hist_full": s.get("x_hist_full"), "x_hat_full": s.get("x_hat_full"),
                "x_gt_full": s.get("x_gt_full"),
                "direction_labels": C.direction_labels(s["x_hat"], s["x_gt"], args.nochange_band),
                "recent_residuals": {"horizon1": packs["perweek"][0]["recent"],
                                     "horizon3": packs["perweek"][1]["recent"]},
                "history_block": packs["history_block"], "baseline_block": packs["combined_baseline"],
                "recent_residual_block": packs["combined_recent"],
                "weekly_text_block": packs["weekly_text_block"],
                "cot": cot, "cot_chars": len(cot), "parsed_directions": parsed,
                "round": args.round,
            }
            if args.round > 1:
                row["momentum_text"] = s.get("_momentum", "")
            C.append_jsonl(gen_path, row); n += 1
    logger.info("FLAT combine → %d combined 2-week rows in %s", n, Path(gen_path).name)


def run_flat(states, splits, pop_map, gemini, out_dir, args, date_from, date_to,
             abstain_ids=None):
    sfx = C.round_suffix(args.round)
    sw_path = out_dir / f"_flat_singleweek{sfx}.jsonl"
    gen_path = out_dir / f"step1_generated{sfx}.jsonl"
    sw_done = {(r["id"], int(r["week_index"]), int(r["candidate_index"]))
               for r in C.iter_jsonl(sw_path) if "week_index" in r}
    sample_by_id = {}
    for state in states:
        cdc_store = args.bundle.store_for(state)
        samples, cache, scale = _prep_state_samples(
            state, splits, C.build_flat_samples, pop_map, cdc_store, args, date_from, date_to)
        if abstain_ids is not None:
            samples = [s for s in samples if s["id"] in abstain_ids]
        if not samples:
            logger.warning("[%s] no flat samples after windowing — skipping.", state)
            continue
        for s in samples:
            s["_packs"] = build_flat_packs(s, cdc_store, cache, scale, args)
            sample_by_id[s["id"]] = s
        logger.info("[%s] flat samples=%d", state, len(samples))
    if args.round > 1:
        # one directional-signal + strength-weighted-verdict summary per date, shared by both
        # of the date's single-week prompts. Flat mode's recent residuals ARE the GT changes
        # at the two horizons (perweek[0]/[1]); everything else matches the chronos2 path.
        hists = {st: args.bundle.histories[st]
                 for st in sorted({s["state"] for s in sample_by_id.values()})
                 if st in args.bundle.histories}
        inputs = {}
        for sid, s in sample_by_id.items():
            st = s["state"]
            eps = (C.build_analog_episodes(hists[st], s["reference_date"])
                   if st in hists else [])
            pw = s["_packs"]["perweek"]
            mom = C.assemble_momentum(
                state=st, reference_date=s["reference_date"],
                recent1=pw[0]["recent"], recent3=pw[1]["recent"], is_flat=True,
                episodes=eps, surveillance=s["_packs"].get("_surveillance"),
                hist=hists.get(st), neighbor_hists=hists, lookback=args.lookback)
            if mom is not None:
                inputs[sid] = mom
        momentum = ensure_momentum(inputs, gemini, out_dir, args)
        for sid, s in sample_by_id.items():
            s["_momentum"] = momentum.get(sid, "")
    pending = [(s, w_idx, ci)
               for s in sample_by_id.values()
               for w_idx in range(len(C.FLAT_PRED_OFFSETS))
               for ci in range(args.k)
               if (s["id"], w_idx, ci) not in sw_done]
    logger.info("STEP1-GEN (flat, round %d): %d single-week candidates (2 weeks × k), "
                "%d-way parallel", args.round, len(pending), args.workers)
    writer = lambda row: C.append_jsonl(sw_path, row) if row else None
    C.run_parallel(pending, lambda it: flat_sw_row(it, gemini, args),
                   args.workers, writer, label="G1",
                   state_of=lambda row: row.get("state") if row else None)
    combine_flat(sw_path, gen_path, sample_by_id, args)


def main():
    args = parse_args()
    C.setup_logging(args.verbose)
    args.provider = C.normalize_provider(args.model)
    args.gen_model = args.gen_model or C.provider_pro_model(args.provider)
    args.momentum_model = args.momentum_model or C.provider_pro_model(args.provider)
    args.nochange_band = C.default_nochange_band(args.mode, args.nochange_band, args.data_mode)
    if (args.data_mode in C.FINANCE_DATA_MODES or args.data_mode in C.COVID_DATA_MODES) \
            and args.mode != "chronos2_full":
        sys.exit(f"--data-mode {args.data_mode} requires --mode chronos2_full (values used "
                 "exactly as preserved in the input CSVs, relative no-change band)")
    states_arg = args.states or C.default_states(args.data_mode)
    states = [s for s in (t.strip() for t in states_arg.split(",")) if s]
    splits = [s for s in (t.strip() for t in args.splits.split(",")) if s]
    if not states:
        sys.exit("--states must list at least one state code")

    args.bundle = C.load_data_bundle(args.data_mode, states, args.data_dir)
    pop_map = args.bundle.pop_map
    gemini = None if args.dry_run else C.make_llm_client(args.provider, args.credentials,
                                   thinking_budget=args.thinking_budget,
                                   thinking_level=args.thinking_level,
                                   use_batch=args.batch)
    if args.batch and args.workers < C.BATCH_WORKERS:
        logger.info("--batch: raising --workers %d -> %d so all pending calls "
                    "land in one batch job", args.workers, C.BATCH_WORKERS)
        args.workers = C.BATCH_WORKERS

    def _open(s): return (s or "").strip().lower() in ("", "none", "null")
    date_from = None if _open(args.sample_date_from) else args.sample_date_from
    date_to   = None if _open(args.sample_date_to)   else args.sample_date_to

    out_dir = C.scratch_run_dir(states, args.mode, args.output_dir, args.scratch_root,
                                provider=args.provider, data_mode=args.data_mode)
    logger.info("run output dir (%s mode, round %d, provider %s): %s",
                args.mode, args.round, args.provider, out_dir)
    abstain_ids = None
    if args.round > 1:
        abstain_ids = load_abstain_ids(out_dir)
        logger.info("ROUND 2: %d round-1 abstain dates to retry with momentum", len(abstain_ids))
        if not abstain_ids:
            logger.info("nothing to retry — round 2 is a no-op.")
            return
    if args.mode == "flat":
        run_flat(states, splits, pop_map, gemini, out_dir, args, date_from, date_to,
                 abstain_ids=abstain_ids)
    else:
        run_chronos2(states, splits, pop_map, gemini, out_dir, args, date_from, date_to,
                     abstain_ids=abstain_ids)
    logger.info("Run complete (Step-1 direction generation, %s mode, round %d).",
                args.mode, args.round)


if __name__ == "__main__":
    main()
