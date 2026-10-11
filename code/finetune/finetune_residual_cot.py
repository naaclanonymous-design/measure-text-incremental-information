"""Residual-only-reasoning ablation finetune (NEW script — never touches existing runs).

Same operator-finetune pipeline as finetune.py (same corpus rows, same train/val/test
composition and skip rules, same prompts, loss plugins, swift training and categorical
evaluation), with ONE change: the assistant TARGET's reasoning is NOT the generated
Step-1/Step-2 CoT. Instead (--reasoning-style):

  residual (default) : Reasoning: <the example's recent_residual_block, verbatim>
                       — the model is trained to "reason" only by restating the recent
                       baseline residuals, then emit the operator (Direction/Size lines).
  none               : no Reasoning section at all — the target is ONLY the
                       Direction/Size lines (direct-answer ablation).

AND, by default, the weekly-report TEXT is removed from the INPUT prompts of every split
(--text drop): the weekly_text_block is replaced by the same stub the notext inference
ablation uses ("(no weekly reports available for this state)"), so both training and
evaluation see numbers only — history, baseline, residuals. This is the full NO-TEXT
finetune arm: no text in the input, no CoT in the supervision.
--text keep retains the reports in the prompts (then the run isolates only the value of
the generated CoT *supervision* while holding the inputs fixed).

Outputs go to a NEW method subdir so nothing collides with existing runs:
  <method>_residualcot_notext/  (default: residual reasoning target, text dropped)
  <method>_residualcot/         (--text keep)
  <method>_nocot_notext/, <method>_nocot/  (--reasoning-style none variants)

Influenza corpus location: finetune.py expects <input-root>/influenza/..., but the
influenza corpora on disk live under <input-root>/influenza_TSFM/...; this script falls
back to influenza_TSFM automatically (and mirrors that dir name on the output side, next
to the existing influenza_TSFM finetune runs). --corpus-dir overrides the corpus dir
explicitly.

All finetune.py flags work unchanged (they are forwarded). Extra flags here:
  --reasoning-style residual|none   (default residual)
  --text drop|keep                  (default drop: remove reports from ALL prompts)
  --corpus-dir PATH                 explicit corpus run dir (containing accepted.jsonl)

Run (influenza, matching the existing scale runs' config for a like-for-like comparison):
  python finetune_residual_cot.py --data-mode influenza --mode chronos2_full \
      --method scale --states GA,NY,NJ --lr 1e-4 --scale-schedule "3:1>0.5:4"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import finetune as ft


# ── extra flags (pre-parsed; the rest is forwarded verbatim to finetune.parse_args) ──
def parse_extra():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--reasoning-style", choices=("residual", "none"), default="residual")
    pre.add_argument("--text", choices=("drop", "keep"), default="drop")
    pre.add_argument("--corpus-dir", type=Path, default=None)
    extra, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return extra


NOTEXT_STUB = "(no weekly reports available for this state)"   # same stub as
# inference_text_ablation.py --variant notext, for consistency across ablations


def residual_reasoning(ex: dict) -> str:
    blk = (ex.get("recent_residual_block") or "").strip()
    return blk


def drop_text(by_split) -> None:
    """Remove the weekly-report text from the INPUT prompts of every split (train, val
    AND test): the model is trained and evaluated on numbers only."""
    n = 0
    for rows in by_split.values():
        for ex in rows:
            ex["weekly_text_block"] = NOTEXT_STUB
            n += 1
    print(f"  [notext] replaced weekly_text_block with stub in {n} examples (all splits)")


def rewrite_cots(by_split, style: str) -> None:
    """Replace every example's target reasoning. Done AFTER load_examples so the split
    composition (including the empty-CoT skip rule) is IDENTICAL to the original runs —
    same rows, different targets. OFFLINE eval rows (covid/finance val+test, built with
    operator None and DUMMY dir/size targets) are left with cot='' so they stay
    inference-eval only, exactly as in the original pipeline — giving them a reasoning
    would make run_one write their bogus targets into val.jsonl and feed swift a
    meaningless val loss for best-checkpoint selection."""
    n = n_skip = 0
    for split, rows in by_split.items():
        for ex in rows:
            if ex.get("operator") is None:      # offline eval row: keep cot=''
                n_skip += 1
                continue
            ex["cot"] = residual_reasoning(ex) if style == "residual" else ""
            n += 1
    print(f"  [{style}] rewrote target reasoning for {n} examples "
          f"({', '.join(f'{k}={len(v)}' for k, v in by_split.items())})"
          + (f"; {n_skip} offline eval rows kept cot-less" if n_skip else ""))


def build_target_text_nocot(ex: dict) -> str:
    dirs = ", ".join(ft.DIR_WORD.get(d, "no change") for d in ex["dir_target"])
    sizes = ", ".join(ex["size_target"])
    return f"{ft.DIRECTION_TAG} {dirs}\n{ft.SIZE_TAG} {sizes}"


def main():
    extra = parse_extra()
    args = ft.parse_args()
    args.scale_start, args.scale_end = ft.parse_schedule(args.scale_schedule)
    if args.nochange_band is None:
        args.nochange_band = (ft.NOCHANGE_ABS_INDEX_BAND if args.data_mode in ft.COVID_MODES
                              else ft.NOCHANGE_REL_BAND if args.mode == "chronos2_full"
                              else ft.NOCHANGE_BAND)
    if (args.data_mode in ft.FINANCE_MODES or args.data_mode in ft.COVID_MODES) \
            and args.mode != "chronos2_full":
        sys.exit(f"--data-mode {args.data_mode} requires --mode chronos2_full")
    states_arg = args.states or ft.DATA_MODE_STATES[args.data_mode]
    states = [s.strip().upper() for s in states_arg.split(",") if s.strip()]
    if not states:
        sys.exit("--states must list at least one state")
    eval_states_arg = (args.eval_states or ft.DATA_MODE_EVAL_STATES.get(args.data_mode)
                       or states_arg)
    eval_states = [s.strip().upper() for s in eval_states_arg.split(",") if s.strip()]
    horizon = ft.horizon_for(args.data_mode, args.mode)
    combo = f"{ft.run_dir_name(states)}-{args.corpus_model.lower()}"

    # corpus dir: explicit --corpus-dir > <input-root>/<data_mode>/... > influenza_TSFM fallback
    data_dir = args.data_mode
    if extra.corpus_dir is not None:
        run_dir = extra.corpus_dir
    else:
        run_dir = args.input_root / data_dir / args.mode / combo
        if not (run_dir / "accepted.jsonl").exists() and args.data_mode == "influenza":
            alt = args.input_root / "influenza_TSFM" / args.mode / combo
            if (alt / "accepted.jsonl").exists():
                data_dir, run_dir = "influenza_TSFM", alt

    style = extra.reasoning_style
    print(f"=== RESIDUAL-CoT ablation finetune  style={style}  text={extra.text}  "
          f"data-mode={args.data_mode}  "
          f"mode={args.mode}  method={args.method}  states={combo}  "
          f"eval-states={','.join(eval_states)}  horizon={horizon}  "
          f"model={Path(args.model).name} ===")
    print(f"input corpus: {run_dir}")

    bins = ft.load_bins(run_dir)
    by_split = ft.load_examples(run_dir, horizon, bins, keep_states=set(states),
                                data_mode=args.data_mode)
    if args.data_mode in ft.FINANCE_MODES:
        orig = ft.load_finance_orig(states)
        ft.to_original_scale(by_split, orig)
        if not by_split.get("val") and not by_split.get("test"):
            by_split.update(ft.build_finance_eval_examples(states, ["val", "test"], horizon,
                                                           orig, args.data_mode))
    elif args.data_mode in ft.COVID_MODES:
        orig = ft.load_covid_orig(sorted(set(states) | set(eval_states)), args.data_mode)
        ft.to_covid_original_scale(by_split, orig, args.data_mode)
        if not by_split.get("val") and not by_split.get("test"):
            by_split.update(ft.build_covid_eval_examples(eval_states, ["val", "test"], horizon,
                                                         orig, args.data_mode))

    # THE ablation: swap every target's reasoning for the residual block (or nothing),
    # and (default) strip the report text from every prompt
    rewrite_cots(by_split, style)
    if style == "none":
        ft.build_target_text = build_target_text_nocot   # target = operator lines only
    if extra.text == "drop":
        drop_text(by_split)

    run_name = f"{Path(args.model).name}_{args.method}_lr{args.lr:g}"
    if args.method == "original":
        run_name += f"_w{args.operator_loss_weight:g}"
    elif args.method == "scale":
        s, e = args.scale_start, args.scale_end
        run_name += f"_sched{s[0]:g}-{s[1]:g}to{e[0]:g}-{e[1]:g}"
    if args.max_new_tokens != ft.MAX_NEW_TOKENS:
        run_name += f"_mnt{args.max_new_tokens}"
    args.class_weights_spec = args.class_weights
    args.class_weights = ft.parse_class_weights(args.class_weights)
    cw_tag = ft.class_weights_tag(args.class_weights)
    if cw_tag:
        run_name += f"_{cw_tag}"
    method_dir = args.method + ("_residualcot" if style == "residual" else "_nocot") \
        + ("_notext" if extra.text == "drop" else "")
    out_dir = args.output_root / data_dir / args.mode / combo / method_dir / run_name
    print(f"output dir: {out_dir}")
    ft.run_one(args, out_dir, by_split, horizon, bins)


if __name__ == "__main__":
    main()
