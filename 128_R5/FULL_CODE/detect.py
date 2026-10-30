"""Official re-run interface: causal detector for one as-of week."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from storepulse.pipeline import load_pack, train_models, predict_records, emerging_flags

def main() -> None:
    ap = argparse.ArgumentParser(description="StorePulse R5 causal emerging-issue detector")
    ap.add_argument("--feedback", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    feedback_path = Path(args.feedback).resolve()
    data_dir = feedback_path.parent
    # The organisers may pass a disguised data folder. Every data reference is loaded from this same folder.
    feedback, labels, taxonomy, stores, _calendar = load_pack(data_dir)
    models = train_models(feedback, labels, taxonomy)
    # The as-of cutoff applies to detector evidence; classifier training uses only public training labels.
    cutoff_monday = pd.Timestamp(args.as_of)
    cutoff_monday = (cutoff_monday - pd.Timedelta(days=cutoff_monday.weekday())).normalize()
    cutoff = cutoff_monday + pd.Timedelta(days=7)
    observed = feedback[pd.to_datetime(feedback["created_at"], errors="coerce") < cutoff].copy()
    predictions = predict_records(observed, models)
    flags = emerging_flags(observed, predictions, stores, as_of=args.as_of, max_flags=3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    flags.to_csv(out, index=False)
    print(f"as-of {cutoff_monday.date()}: {len(observed):,} feedback rows available; wrote {len(flags)} flags to {out}")

if __name__ == "__main__":
    main()
