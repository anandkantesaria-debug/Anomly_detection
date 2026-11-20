"""Health Indicator, Early-Warning Alert Engine, and RUL Prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

METADATA_COLS = {
    "split",
    "bearing",
    "file_name",
    "snapshot_idx",
    "elapsed_min",
    "rul_min",
    "rul_pct",
}

HI_DRIVER_COLS = [
    "h_rms_rel",
    "v_rms_rel",
    "h_rms_ewma",
    "v_rms_ewma",
    "h_kurtosis_ewma",
    "v_kurtosis_ewma",
    "h_env_rms_ewma",
    "v_env_rms_ewma",
    "h_peak",
    "v_peak",
]


@dataclass
class AlertConfig:
    """Thresholds and persistence hysteresis for industrial bearing failure flagging."""

    warning_hi: float = 0.35
    critical_hi: float = 0.70
    warning_rul_pct: float = 0.30
    critical_rul_pct: float = 0.12
    persistence: int = 3


class BearingPHMPipeline:
    """Dual-output Prognostics & Health Management model for PRONOSTIA bearings.

    Outputs:
      1. `health_index` in [0, 1]: 0 = healthy baseline, 1 = severe degradation / failure.
      2. `pred_rul_pct` in [0, 1] and `pred_rul_min`: predicted Remaining Useful Life.
      3. `alert_state`: 'HEALTHY', 'WARNING', or 'CRITICAL' with persistence hysteresis.
    """

    def __init__(self, alert_config: AlertConfig | None = None) -> None:
        self.alert_config = alert_config or AlertConfig()
        self.feature_cols: list[str] = []
        self.hi_cols: list[str] = []

        self.hi_scaler = RobustScaler()
        self.hi_pca = PCA(n_components=3, random_state=42)
        self.iso_forest = IsolationForest(n_estimators=150, contamination=0.05, random_state=42)
        self.hi_p05: float = 0.0
        self.hi_p95: float = 1.0

        self.rul_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    HistGradientBoostingRegressor(
                        max_depth=5,
                        learning_rate=0.05,
                        max_iter=250,
                        min_samples_leaf=15,
                        random_state=42,
                    ),
                ),
            ]
        )
        self.median_life_by_cond: dict[int, float] = {}

    def _select_feature_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in METADATA_COLS]

    def _raw_health_score(self, df: pd.DataFrame) -> np.ndarray:
        X = self.hi_scaler.transform(df[self.hi_cols].to_numpy(dtype=np.float64))
        z = self.hi_pca.transform(X)
        # Distance from healthy origin in PCA space + IsolationForest anomaly score
        pca_dist = np.linalg.norm(z, axis=1)
        iso_anom = -self.iso_forest.score_samples(X)
        return 0.65 * pca_dist + 0.35 * iso_anom

    def fit(self, train_df: pd.DataFrame) -> "BearingPHMPipeline":
        """Train unsupervised Health Indicator baseline and supervised RUL regressor."""
        self.feature_cols = self._select_feature_cols(train_df)
        self.hi_cols = [c for c in HI_DRIVER_COLS if c in train_df.columns]

        # Fit healthy baseline on the first 20% of each training bearing's lifecycle
        healthy_mask = train_df["rul_pct"] >= 0.80
        healthy_df = train_df.loc[healthy_mask, self.hi_cols]

        X_healthy = self.hi_scaler.fit_transform(healthy_df.to_numpy(dtype=np.float64))
        self.hi_pca.fit(X_healthy)
        self.iso_forest.fit(X_healthy)

        # Calibrate HI scaling across the full training trajectories
        raw_train = self._raw_health_score(train_df)
        self.hi_p05 = float(np.percentile(raw_train, 5))
        self.hi_p95 = float(np.percentile(raw_train, 97))

        # Fit RUL % regressor on all engineered features
        X_train = train_df[self.feature_cols].to_numpy(dtype=np.float64)
        y_train = train_df["rul_pct"].to_numpy(dtype=np.float64)
        self.rul_model.fit(X_train, y_train)

        # Store median total life (in minutes) per operating condition for minute-level RUL projection
        life_per_bearing = (
            train_df.groupby(["condition", "bearing"])["elapsed_min"].max().reset_index()
        )
        self.median_life_by_cond = (
            life_per_bearing.groupby("condition")["elapsed_min"].median().to_dict()
        )
        return self

    def predict_bearing(self, bearing_df: pd.DataFrame) -> pd.DataFrame:
        """Score a single bearing's trajectory and generate Health Index, RUL, and Alert flags."""
        out = bearing_df.copy().sort_values("snapshot_idx").reset_index(drop=True)

        # 1. Continuous Health Index in [0, 1] with EWMA smoothing
        raw_hi = self._raw_health_score(out)
        norm_hi = (raw_hi - self.hi_p05) / max(1e-9, self.hi_p95 - self.hi_p05)
        norm_hi = np.clip(norm_hi, 0.0, 1.0)
        out["health_index_raw"] = norm_hi
        out["health_index"] = pd.Series(norm_hi).ewm(span=8, adjust=False).mean().to_numpy()

        # 2. Predicted RUL (% and minutes)
        X = out[self.feature_cols].to_numpy(dtype=np.float64)
        pred_rul_pct = np.clip(self.rul_model.predict(X), 0.0, 1.0)
        out["pred_rul_pct"] = pd.Series(pred_rul_pct).ewm(span=6, adjust=False).mean().to_numpy()

        cond_id = int(out["condition"].iloc[0])
        nominal_life = self.median_life_by_cond.get(cond_id, float(out["elapsed_min"].max()))
        # Blend elapsed-based life estimate with nominal condition life
        est_total_life = np.maximum(out["elapsed_min"] / np.maximum(1.0 - out["pred_rul_pct"], 0.05), nominal_life)
        est_total_life = np.minimum(est_total_life, nominal_life * 2.5)
        out["pred_rul_min"] = out["pred_rul_pct"] * est_total_life

        # 3. Alert state machine with persistence hysteresis
        cfg = self.alert_config
        states: list[str] = []
        warn_streak = 0
        crit_streak = 0
        latched_state = "HEALTHY"

        for hi, rul_p in zip(out["health_index"], out["pred_rul_pct"]):
            is_crit = (hi >= cfg.critical_hi) or (rul_p <= cfg.critical_rul_pct)
            is_warn = (hi >= cfg.warning_hi) or (rul_p <= cfg.warning_rul_pct)

            crit_streak = crit_streak + 1 if is_crit else 0
            warn_streak = warn_streak + 1 if is_warn else 0

            if crit_streak >= cfg.persistence:
                latched_state = "CRITICAL"
            elif warn_streak >= cfg.persistence and latched_state != "CRITICAL":
                latched_state = "WARNING"

            states.append(latched_state)

        out["alert_state"] = states
        out["failure_flag"] = out["alert_state"].isin(["WARNING", "CRITICAL"])
        return out

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict across multiple bearings."""
        results = [
            self.predict_bearing(bdf)
            for _, bdf in df.groupby(["split", "bearing"], sort=False)
        ]
        return pd.concat(results, ignore_index=True)


def evaluate_predictions(pred_df: pd.DataFrame) -> dict[str, Any]:
    """Compute RUL regression metrics and early-warning lead time per bearing."""
    rul_pct_mae = float(mean_absolute_error(pred_df["rul_pct"], pred_df["pred_rul_pct"]))
    rul_pct_rmse = float(root_mean_squared_error(pred_df["rul_pct"], pred_df["pred_rul_pct"]))
    rul_min_mae = float(mean_absolute_error(pred_df["rul_min"], pred_df["pred_rul_min"]))

    per_bearing: list[dict[str, Any]] = []
    for (split_name, bearing), bdf in pred_df.groupby(["split", "bearing"]):
        total_min = float(bdf["elapsed_min"].max())
        warn_rows = bdf[bdf["alert_state"] == "WARNING"]
        crit_rows = bdf[bdf["alert_state"] == "CRITICAL"]

        warn_lead_min = float(warn_rows["rul_min"].iloc[0]) if not warn_rows.empty else None
        crit_lead_min = float(crit_rows["rul_min"].iloc[0]) if not crit_rows.empty else None

        per_bearing.append(
            {
                "split": split_name,
                "bearing": bearing,
                "total_life_min": round(total_min, 2),
                "rul_pct_mae": round(float(mean_absolute_error(bdf["rul_pct"], bdf["pred_rul_pct"])), 4),
                "warning_lead_min": round(warn_lead_min, 2) if warn_lead_min is not None else None,
                "critical_lead_min": round(crit_lead_min, 2) if crit_lead_min is not None else None,
            }
        )

    return {
        "overall": {
            "rul_pct_mae": round(rul_pct_mae, 4),
            "rul_pct_rmse": round(rul_pct_rmse, 4),
            "rul_min_mae": round(rul_min_mae, 2),
        },
        "per_bearing": per_bearing,
    }
