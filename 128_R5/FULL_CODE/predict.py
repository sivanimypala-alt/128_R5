"""Train from the public labelled sample and create official-format R5 predictions."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
from storepulse.pipeline import load_pack, train_models, predict_records, emerging_flags

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "submission"))
    ap.add_argument("--max-flags", type=int, default=3)
    args = ap.parse_args()
    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    feedback, labels, taxonomy, stores, _calendar = load_pack(data_dir)
    models = train_models(feedback, labels, taxonomy)
    predictions = predict_records(feedback, models)
    predictions.to_csv(out_dir / "R5_predictions.csv", index=False)
    flags = emerging_flags(feedback, predictions, stores, max_flags=args.max_flags)
    flags.to_csv(out_dir / "R5_emerging_flags.csv", index=False)
    (out_dir / "detector_cmd.txt").write_text(
        "python FULL_CODE/detect.py --feedback {feedback} --as-of {asof} --out {out}\n", encoding="utf-8")
    print(f"Training records: {models['train_rows']:,}")
    print(f"Predictions: {len(predictions):,} -> {out_dir / 'R5_predictions.csv'}")
    print(f"Emerging flags: {len(flags):,} -> {out_dir / 'R5_emerging_flags.csv'}")
    print("No hidden labels were used. Review evaluator metrics before claiming performance.")

if __name__ == "__main__":
    main()
