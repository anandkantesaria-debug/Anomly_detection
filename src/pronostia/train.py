"""End-to-end training & evaluation CLI for the PRONOSTIA bearing monitor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from pronostia.features import build_feature_dataset
from pronostia.model import BearingPHMPipeline, evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features and train PRONOSTIA PHM models.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Raw PRONOSTIA dataset root.")
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed"), help="Directory for extracted features."
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Directory for model artifacts.")
    args = parser.parse_args()

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    features_df = build_feature_dataset(args.raw_dir)
    features_path = args.processed_dir / "features.parquet"
    features_df.to_parquet(features_path, index=False)
    print(f"[OK] Saved {len(features_df):,} snapshots across {features_df['bearing'].nunique()} bearings -> {features_path}")

    # Strict bearing-level train/test split (no snapshot leakage)
    train_df = features_df[features_df["split"] == "Learning_set"].copy()
    test_df = features_df[features_df["split"] != "Learning_set"].copy()

    if test_df.empty:
        # Fallback if only Learning_set was downloaded: hold out Bearing*_2 for testing
        is_test = train_df["bearing"].str.endswith("_2")
        test_df = train_df[is_test].copy()
        train_df = train_df[~is_test].copy()

    print(
        f"[TRAIN] Train bearings: {sorted(train_df['bearing'].unique())} | "
        f"Test bearings: {sorted(test_df['bearing'].unique())}"
    )

    pipeline = BearingPHMPipeline()
    pipeline.fit(train_df)

    scored_all = pipeline.predict(features_df)
    scored_path = args.processed_dir / "predictions.parquet"
    scored_all.to_parquet(scored_path, index=False)

    test_preds = scored_all[scored_all["bearing"].isin(test_df["bearing"].unique())]
    metrics = evaluate_predictions(test_preds)

    model_path = args.models_dir / "bearing_phm_pipeline.joblib"
    metrics_path = args.models_dir / "metrics.json"
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== Test Set Evaluation ===")
    print(json.dumps(metrics, indent=2))
    print(f"\n[OK] Model saved -> {model_path}")
    print(f"[OK] Predictions saved -> {scored_path}")


if __name__ == "__main__":
    main()
