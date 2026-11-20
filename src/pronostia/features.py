"""Signal processing and feature extraction for PRONOSTIA (25.6 kHz) vibration snapshots."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats

FS_HZ = 25600.0
SNAPSHOT_INTERVAL_SEC = 10.0

OPERATING_CONDITIONS = {
    1: {"rpm": 1800.0, "load_n": 4000.0},
    2: {"rpm": 1650.0, "load_n": 4200.0},
    3: {"rpm": 1500.0, "load_n": 5000.0},
}

# Frequency bands (Hz) up to Nyquist (12,800 Hz)
FREQ_BANDS = [
    ("band_0_2k", 0.0, 2000.0),
    ("band_2_5k", 2000.0, 5000.0),
    ("band_5_9k", 5000.0, 9000.0),
    ("band_9_12k", 9000.0, 12800.0),
]


def load_snapshot_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load horizontal and vertical accelerometer arrays from a PRONOSTIA acc_*.csv file.

    Handles both comma-separated and semicolon-separated files (e.g., Bearing1_4).
    """
    first_line = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    sep = ";" if ";" in first_line else ","
    data = np.loadtxt(csv_path, delimiter=sep, usecols=(4, 5), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1]


def extract_axis_features(x: np.ndarray, prefix: str, fs: float = FS_HZ) -> dict[str, float]:
    """Compute time-domain, frequency-domain, and Hilbert envelope features for a 1D vibration signal."""
    eps = 1e-12
    x = x - np.mean(x)
    abs_x = np.abs(x)

    rms = float(np.sqrt(np.mean(x * x) + eps))
    peak = float(np.max(abs_x))
    p2p = float(np.max(x) - np.min(x))
    mean_abs = float(np.mean(abs_x) + eps)

    crest_factor = peak / (rms + eps)
    impulse_factor = peak / mean_abs
    shape_factor = rms / mean_abs
    kurt = float(stats.kurtosis(x, fisher=False))
    skew = float(stats.skew(x))
    energy = float(np.mean(x * x))

    # Frequency-domain (Welch PSD)
    nperseg = min(512, len(x))
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg)
    psd_sum = float(np.sum(psd) + eps)
    psd_norm = psd / psd_sum

    spectral_centroid = float(np.sum(freqs * psd_norm))
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + eps)))
    dominant_freq = float(freqs[int(np.argmax(psd))])

    feats: dict[str, float] = {
        f"{prefix}_rms": rms,
        f"{prefix}_peak": peak,
        f"{prefix}_p2p": p2p,
        f"{prefix}_crest": crest_factor,
        f"{prefix}_impulse": impulse_factor,
        f"{prefix}_shape": shape_factor,
        f"{prefix}_kurtosis": kurt,
        f"{prefix}_skew": skew,
        f"{prefix}_energy": energy,
        f"{prefix}_spec_centroid": spectral_centroid,
        f"{prefix}_spec_entropy": spectral_entropy,
        f"{prefix}_dom_freq": dominant_freq,
    }

    for band_name, f_lo, f_hi in FREQ_BANDS:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        feats[f"{prefix}_{band_name}_ratio"] = float(np.sum(psd[mask]) / psd_sum)

    # Hilbert envelope features (sensitive to periodic bearing raceway impacts)
    analytic = signal.hilbert(x)
    env = np.abs(analytic)
    env_centered = env - np.mean(env)
    feats[f"{prefix}_env_rms"] = float(np.sqrt(np.mean(env_centered * env_centered) + eps))
    feats[f"{prefix}_env_kurtosis"] = float(stats.kurtosis(env, fisher=False))

    return feats


def parse_bearing_condition(bearing_name: str) -> int:
    """Extract operating condition (1, 2, or 3) from bearing folder name like 'Bearing1_3'."""
    match = re.search(r"Bearing(\d+)_", bearing_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1


def extract_bearing_features(bearing_dir: Path, split_name: str) -> pd.DataFrame:
    """Extract snapshot-by-snapshot vibration features for a single bearing run-to-failure folder."""
    csv_files = sorted(bearing_dir.glob("acc_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No acc_*.csv files found in {bearing_dir}")

    bearing_name = bearing_dir.name
    cond_id = parse_bearing_condition(bearing_name)
    cond = OPERATING_CONDITIONS.get(cond_id, OPERATING_CONDITIONS[1])
    total_snapshots = len(csv_files)

    rows: list[dict[str, float | int | str]] = []
    for idx, csv_path in enumerate(csv_files):
        horiz, vert = load_snapshot_csv(csv_path)
        row: dict[str, float | int | str] = {
            "split": split_name,
            "bearing": bearing_name,
            "condition": cond_id,
            "rpm": cond["rpm"],
            "load_n": cond["load_n"],
            "snapshot_idx": idx,
            "file_name": csv_path.name,
            "elapsed_min": (idx * SNAPSHOT_INTERVAL_SEC) / 60.0,
            "rul_min": ((total_snapshots - 1 - idx) * SNAPSHOT_INTERVAL_SEC) / 60.0,
            "rul_pct": (total_snapshots - 1 - idx) / max(1, total_snapshots - 1),
        }
        row.update(extract_axis_features(horiz, "h"))
        row.update(extract_axis_features(vert, "v"))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add EWMA trend and rolling slope features to suppress transient spikes
    for col in ("h_rms", "v_rms", "h_kurtosis", "v_kurtosis", "h_env_rms", "v_env_rms"):
        ewma = df[col].ewm(span=10, adjust=False).mean()
        df[f"{col}_ewma"] = ewma
        df[f"{col}_slope"] = ewma.diff(periods=5).fillna(0.0)
        # Baseline-normalized ratio relative to first 10% healthy run-in
        baseline_n = max(5, int(0.10 * len(df)))
        baseline_val = float(df[col].iloc[:baseline_n].median()) + 1e-9
        df[f"{col}_rel"] = df[col] / baseline_val

    return df


def discover_bearing_dirs(raw_dir: Path) -> list[tuple[str, Path]]:
    """Find all bearing directories under Learning_set / Full_Test_Set / Test_set."""
    discovered: list[tuple[str, Path]] = []
    for split_name in ("Learning_set", "Full_Test_Set", "Test_set"):
        split_path = raw_dir / split_name
        if not split_path.exists():
            continue
        for child in sorted(split_path.iterdir()):
            if child.is_dir() and any(child.glob("acc_*.csv")):
                discovered.append((split_name, child))
    return discovered


def build_feature_dataset(raw_dir: Path) -> pd.DataFrame:
    """Extract and concatenate features for all bearings found in raw_dir."""
    bearing_dirs = discover_bearing_dirs(raw_dir)
    if not bearing_dirs:
        raise FileNotFoundError(
            f"No PRONOSTIA bearing folders found in {raw_dir}. "
            "Run `uv run pronostia-download` or `uv run pronostia-download --synthetic` first."
        )

    frames = []
    for split_name, bdir in bearing_dirs:
        print(f"[FEAT] Extracting {split_name}/{bdir.name} ...")
        frames.append(extract_bearing_features(bdir, split_name))

    return pd.concat(frames, ignore_index=True)
