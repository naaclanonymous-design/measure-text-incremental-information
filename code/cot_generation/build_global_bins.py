"""Build the GLOBAL size-class bins for a run, pooled across every state's residuals.

This is optional: step2_generate.py builds (and caches) the run bins automatically on
first run. Use this to (re)build and inspect them deliberately, before Step-2 generation.

The bins are pooled over MOVING cases (|r| > ε) across ALL states present in this run's
step1_filtered.jsonl (the merged two-round view when the momentum retry has run), written
to the run's bins.json. Every state and every Step-2 stage of the run then uses the SAME
definitions, so the operator's "large increase" means the same magnitude in every state.
chronos2/flat: |r| quantile classes (per-10,000 rate units). chronos2_full: RELATIVE
%-of-baseline QUANTILE classes over the pooled moving change-proportions, so each
class holds ~the same share of moving cases (~25% each for 4 classes).

Run (after step1_filter has been run for the same --states):
    python build_global_bins.py --states GA,NJ,NY
    python build_global_bins.py --states GA,NJ,NY --rebuild      # re-pool and overwrite
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-mode", type=str, default=C.DATA_MODE_DEFAULT,
                   choices=C.DATA_MODES,
                   help="input-data source (influenza | finance_stat | finance_TSFM); selects the "
                        "matching output subdirectory")
    p.add_argument("--states", type=str, default=None,
                   help="the SAME states used for the run (selects the run output dir)")
    p.add_argument("--mode", type=str, default=C.MODE_DEFAULT, choices=C.MODES)
    p.add_argument("--bin-names", type=str, default=",".join(C.BIN_NAMES_DEFAULT))
    p.add_argument("--bins-split", type=str, default="train",
                   help="pool |r| from this split only (avoids test leakage); '' = all")
    p.add_argument("--nochange-band", type=float, default=None,
                   help="no-change band ε; default 0.01 per-10,000 (chronos2/flat), 0.10 "
                        "RELATIVE fraction of the baseline (chronos2_full), or 20 ABSOLUTE "
                        "index points (covid_TSFM)")
    p.add_argument("--model", type=str, default=C.PROVIDER_DEFAULT,
                   help="model family of the run (gemini|GPT) — selects the run output dir")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--output-dir", type=Path, default=C.OUTPUT_DIR)
    p.add_argument("--scratch-root", type=Path, default=C.SCRATCH_OUTPUT_ROOT)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if getattr(args, "prompt_dir", None) is None:      # prompts/covid for covid modes, prompts/influenza otherwise
        args.prompt_dir = C.prompt_dir_for(getattr(args, "data_mode", "influenza"))
    return args


def main():
    args = parse_args()
    C.setup_logging(args.verbose)
    states = [s for s in (t.strip() for t in
                      (args.states or C.default_states(args.data_mode)).split(",")) if s]
    out_dir = C.scratch_run_dir(states, args.mode, args.output_dir, args.scratch_root,
                                provider=C.normalize_provider(args.model),
                                data_mode=args.data_mode)
    names = [s.strip() for s in args.bin_names.split(",") if s.strip()]
    bins_split = None if args.bins_split == "" else args.bins_split
    args.nochange_band = C.default_nochange_band(args.mode, args.nochange_band, args.data_mode)
    bins = C.build_or_load_run_bins(out_dir, names=names, eps=args.nochange_band,
                                    bins_split=bins_split, rebuild=args.rebuild,
                                    relative=C.band_is_relative(args.mode, args.data_mode),
                                    abs_index=(args.data_mode in C.COVID_DATA_MODES))
    print("\n=== GLOBAL run bins ===")
    print("run dir      :", out_dir)
    print("scope        :", bins.get("scope"))
    print("n_moving     :", bins["n_moving"], " epsilon:", bins["epsilon"])
    print("\n" + C.fmt_bin_definitions(bins))
    print("\nlower edges (cold-start m_init):", [round(x, 4) for x in bins["lower_edges"]])


if __name__ == "__main__":
    main()
