#!/usr/bin/env python3
"""Build the "complete" chronos2_full finetune corpus (no generation rerun needed).

For each date, keep ALL passing Step-1 direction CoTs from filter1 AND regenerate-filter1
(round-1 + round-2), taken from the merged summary rows' `passing_cots`. Each passing CoT
becomes its own training row via `step1_basis_cot`; the date's single passing Step-2 size CoT
(`step2_size_cot`) is kept as-is, so the finetune loader concatenates them (step1 + step2) at
load time — exactly the current chronos2_full format. Abstain dates keep ONE line.

Only the TRAIN split is expanded (more move/no_change signal, far less abstain dominance —
the fix for the "always no change" collapse). VAL and TEST stay one-example-per-date so
evaluation remains per-date and directly comparable to the standard runs. Adjust EXPAND_SPLITS
to change that.

Output is a drop-in `accepted.jsonl` (+ `bins.json`) in the SAME schema the finetune script
reads, written next to the source run (default):
    <PAPER_OUTPUT_ROOT>/cot_corpora/influenza/chronos2_full/GA_NY_NJ-gemini-complete/

Input: the full 6-stage pipeline run (code/cot_generation/run_cot_pipeline.sh influenza), whose
accepted.jsonl AND step1_filtered_merged.jsonl (the passing_cots of every origin) are needed; the
released data/cot_corpora/influenza/GA_NY_NJ-gemini/ holds only accepted.jsonl.
The released data/cot_corpora/influenza/GA_NY_NJ-gemini-final/ is exactly this script's output
for the pipeline run behind the paper (866 rows).

Run:  python build_complete_corpus.py                       # default source / output
      python build_complete_corpus.py --src <run dir> --dst <out dir>
Then: finetune.py --data-mode influenza --mode chronos2_full --method scale \
        --corpus-model gemini-complete --states GA,NY,NJ ...   (see code/finetune/*.sbatch)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]          # <paper-content>/code
for _d in ("cot_generation", "finetune", "zero_shot"):
    sys.path.insert(0, str(_CODE_ROOT / _d))
sys.path.insert(0, str(_CODE_ROOT))                         # paper_paths.py
import paper_paths as P                                     # noqa: E402
AUG_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"   # the prompts of this stage

SRC_ROOT = P.COT_OUTPUT_DIR / "influenza" / "chronos2_full"     # the 6-stage pipeline output
DST_ROOT = SRC_ROOT                                              # <combo>-complete/ next to the source
COMBOS = ["GA_NY_NJ-gemini"]                                     # the run behind the reported result
EXPAND_SPLITS = {"train"}   # only these splits get all passing CoTs; others stay 1-per-date


def load_passing_cots(merged_path: Path) -> dict:
    """id -> list of passing filter1 CoT dicts (the merged summary rows' passing_cots),
    covering round-1 passers and, for rescued dates, round-2 passers."""
    out = {}
    with open(merged_path) as f:
        for line in f:
            r = json.loads(line)
            if int(r.get("candidate_index", 0)) == -1:      # date-summary row
                out[r["id"]] = r.get("passing_cots") or []
    return out


def build(combo: str, src: Path | None = None, dst: Path | None = None) -> None:
    src = src or SRC_ROOT / combo
    dst = dst or DST_ROOT / f"{combo}-complete"
    acc_path = src / "accepted.jsonl"
    merged_path = src / "step1_filtered_merged.jsonl"
    if not acc_path.exists() or not merged_path.exists():
        raise FileNotFoundError(f"missing source files under {src}")
    dst.mkdir(parents=True, exist_ok=True)

    acc_rows = [json.loads(l) for l in open(acc_path)]
    passing = load_passing_cots(merged_path)

    out_rows = []
    # per-split example tallies by category (for the printed summary)
    tally = {s: Counter() for s in ("train", "val", "test")}
    n_expanded_dates = 0

    for r in acc_rows:
        st = r["date_status"]
        split = r.get("split")
        rid = r["id"]
        expand = (split in EXPAND_SPLITS) and (st in ("move", "no_change"))

        if not expand:
            # val/test dates, and every abstain date: one line, unchanged
            out_rows.append(r)
            if split in tally:
                tally[split][st] += 1
            continue

        pcs = passing.get(rid) or []
        if not pcs:
            # safety fallback: no passing_cots recorded -> keep the single basis row
            out_rows.append(r)
            tally[split][st] += 1
            continue

        n_expanded_dates += 1
        for pc in pcs:
            nr = dict(r)                                     # copy targets/blocks/step2_size_cot/bins/...
            nr["step1_basis_cot"] = pc.get("cot") or r.get("step1_basis_cot", "")
            ci = int(pc.get("candidate_index", 0))
            nr["id"] = f"{rid}__c{ci}"                       # unique per expanded CoT
            nr["source_id"] = rid                            # provenance back to the origin date
            nr["basis_candidate_index"] = ci
            nr["basis_directions"] = pc.get("parsed_directions", r.get("basis_directions"))
            out_rows.append(nr)
            tally[split][st] += 1

    # write the drop-in corpus
    with open(dst / "accepted.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    if (src / "bins.json").exists():
        shutil.copy(src / "bins.json", dst / "bins.json")

    # summary
    print(f"\n[{combo}] -> {dst}")
    print(f"  expanded {n_expanded_dates} train move/no_change dates; total rows {len(out_rows)} "
          f"(was {len(acc_rows)} one-per-date)")
    for split in ("train", "val", "test"):
        c = tally[split]
        n = sum(c.values())
        if n:
            print(f"  {split:5s}: move={c['move']:4d}  no_change={c['no_change']:3d}  "
                  f"abstain={c['abstain']:4d}  total={n:4d}  "
                  f"({100*c['move']//n}/{100*c['no_change']//n}/{100*c['abstain']//n} %%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=None,
                    help=f"pipeline run dir holding accepted.jsonl + step1_filtered_merged.jsonl (default {SRC_ROOT}/<combo>)")
    ap.add_argument("--dst", type=Path, default=None, help="output dir (default <src>-complete)")
    args = ap.parse_args()
    if args.src is not None:
        build(args.src.name, args.src, args.dst or args.src.parent / f"{args.src.name}-complete")
        return
    for combo in COMBOS:
        build(combo, dst=args.dst)


if __name__ == "__main__":
    main()
