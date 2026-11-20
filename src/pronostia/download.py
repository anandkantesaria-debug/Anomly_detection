"""Download the PRONOSTIA (IEEE PHM 2012) dataset or generate a synthetic PRONOSTIA-format dataset."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO_URL = "https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset.git"

# Operating conditions from the PRONOSTIA (FEMTO-ST) platform
OPERATING_CONDITIONS = {
    1: {"rpm": 1800, "load_n": 4000, "shaft_hz": 30.0},
    2: {"rpm": 1650, "load_n": 4200, "shaft_hz": 27.5},
    3: {"rpm": 1500, "load_n": 5000, "shaft_hz": 25.0},
}

DEFAULT_SYNTHETIC_BEARINGS = {
    "Learning_set": [
        ("Bearing1_1", 1, 140, 101),
        ("Bearing1_2", 1, 110, 102),
        ("Bearing2_1", 2, 130, 201),
        ("Bearing2_2", 2, 105, 202),
        ("Bearing3_1", 3, 95, 301),
        ("Bearing3_2", 3, 120, 302),
    ],
    "Full_Test_Set": [
        ("Bearing1_3", 1, 125, 103),
        ("Bearing2_3", 2, 115, 203),
        ("Bearing3_3", 3, 100, 303),
    ],
}


def download_pronostia(output_dir: Path, learning_only: bool = False) -> None:
    """Clone the IEEE PHM 2012 PRONOSTIA dataset from GitHub."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "Learning_set").exists():
        print(f"[INFO] Dataset already exists at {output_dir}")
        return

    git_bin = shutil.which("git")
    if not git_bin:
        raise RuntimeError("git is required to download the dataset from GitHub.")

    print(f"[INFO] Cloning PRONOSTIA dataset from {REPO_URL} into {output_dir} ...")
    if learning_only:
        subprocess.run(
            [git_bin, "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, str(output_dir)],
            check=True,
        )
        subprocess.run(
            [git_bin, "-C", str(output_dir), "sparse-checkout", "set", "Learning_set"],
            check=True,
        )
    else:
        subprocess.run([git_bin, "clone", "--depth", "1", REPO_URL, str(output_dir)], check=True)

    print(f"[OK] PRONOSTIA dataset ready at {output_dir}")


def _simulate_snapshot(
    t: np.ndarray,
    progress: float,
    shaft_hz: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one 0.1s (2,560-sample) PRONOSTIA vibration snapshot at normalized life progress in [0, 1]."""
    # Healthy baseline: shaft fundamental + 2x harmonic + broadband sensor noise
    base_h = 0.28 * np.sin(2 * np.pi * shaft_hz * t) + 0.12 * np.sin(4 * np.pi * shaft_hz * t)
    base_v = 0.24 * np.cos(2 * np.pi * shaft_hz * t) + 0.10 * np.sin(4 * np.pi * shaft_hz * t)
    noise_h = rng.normal(0.0, 0.35, size=t.shape)
    noise_v = rng.normal(0.0, 0.32, size=t.shape)

    # Degradation onset typically starts around 55%-70% of life and grows exponentially
    onset = 0.58
    if progress <= onset:
        deg = 0.02 * (progress / onset)
    else:
        u = (progress - onset) / (1.0 - onset)
        deg = 0.02 + (np.exp(3.4 * u) - 1.0) / (np.exp(3.4) - 1.0)

    # Bearing outer-race fault frequency (~3.58 * shaft_hz) modulating a 3.2 kHz structural resonance
    bpfo = 3.58 * shaft_hz
    carrier = np.sin(2 * np.pi * 3200.0 * t)
    impulses = np.maximum(0.0, np.sin(2 * np.pi * bpfo * t)) ** 6
    fault_h = deg * 11.5 * impulses * carrier
    fault_v = deg * 9.5 * impulses * np.cos(2 * np.pi * 3200.0 * t)

    # Random high-frequency micro-spalling shocks in late degradation
    shock_h = np.zeros_like(t)
    shock_v = np.zeros_like(t)
    if deg > 0.25:
        n_shocks = int(2 + 12 * deg)
        idx = rng.integers(0, len(t), size=n_shocks)
        shock_h[idx] = rng.normal(0.0, 6.5 * deg, size=n_shocks)
        shock_v[idx] = rng.normal(0.0, 5.5 * deg, size=n_shocks)

    horiz = base_h + (1.0 + 3.0 * deg) * noise_h + fault_h + shock_h
    vert = base_v + (1.0 + 2.6 * deg) * noise_v + fault_v + shock_v
    return horiz.astype(np.float32), vert.astype(np.float32)


def generate_synthetic_pronostia(output_dir: Path, samples_per_snapshot: int = 2560) -> None:
    """Create a realistic PRONOSTIA-format dataset locally for rapid prototyping and testing."""
    fs = 25600.0
    t = np.arange(samples_per_snapshot, dtype=np.float64) / fs
    microseconds = np.round(t * 1e6).astype(int)

    for split_name, bearings in DEFAULT_SYNTHETIC_BEARINGS.items():
        for bearing_name, cond_id, n_files, seed in bearings:
            bdir = output_dir / split_name / bearing_name
            bdir.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(seed)
            shaft_hz = OPERATING_CONDITIONS[cond_id]["shaft_hz"]

            for idx in range(1, n_files + 1):
                progress = (idx - 1) / max(1, n_files - 1)
                horiz, vert = _simulate_snapshot(t, progress, shaft_hz, rng)

                total_sec = (idx - 1) * 10
                hour = total_sec // 3600
                minute = (total_sec % 3600) // 60
                second = total_sec % 60

                df = pd.DataFrame(
                    {
                        0: hour,
                        1: minute,
                        2: second,
                        3: microseconds,
                        4: np.round(horiz, 5),
                        5: np.round(vert, 5),
                    }
                )
                df.to_csv(bdir / f"acc_{idx:05d}.csv", header=False, index=False)

            print(f"[SYNTH] Generated {split_name}/{bearing_name} ({n_files} snapshots)")

    print(f"[OK] Synthetic PRONOSTIA dataset ready at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or generate the PRONOSTIA (PHM 2012) dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Target directory.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a realistic local PRONOSTIA-format synthetic dataset instead of cloning GitHub.",
    )
    parser.add_argument(
        "--learning-only",
        action="store_true",
        help="When cloning from GitHub, only fetch Learning_set (faster, smaller download).",
    )
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_pronostia(args.output_dir)
    else:
        download_pronostia(args.output_dir, learning_only=args.learning_only)


if __name__ == "__main__":
    main()
