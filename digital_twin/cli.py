"""Command-line entry point: generate a labelled synthetic vibration dataset (Parquet)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from digital_twin.generator import generate_vibration
from services.common.schemas import FaultType, SimulationConfig


def build_dataset(params: dict[str, Any]) -> pd.DataFrame:
    """Generate one row per window: metadata, label and raw samples."""
    rng = np.random.default_rng(int(params["seed"]))
    rows: list[dict[str, Any]] = []

    for fault in FaultType:
        for index in range(int(params["windows_per_class"])):
            speed = float(rng.uniform(params["shaft_speed_hz_min"], params["shaft_speed_hz_max"]))
            severity = (
                0.0
                if fault is FaultType.HEALTHY
                else float(rng.uniform(params["severity_min"], params["severity_max"]))
            )
            config = SimulationConfig(
                asset_id=f"synthetic-{fault.value}-{index:04d}",
                fault=fault,
                severity=severity,
                shaft_speed_hz=speed,
                sampling_rate_hz=int(params["sampling_rate_hz"]),
                duration_s=float(params["duration_s"]),
                noise_std=float(params["noise_std"]),
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            window = generate_vibration(config)
            rows.append(
                {
                    "asset_id": window.asset_id,
                    "label": window.label.value,
                    "severity": window.severity,
                    "shaft_speed_hz": window.shaft_speed_hz,
                    "sampling_rate_hz": window.sampling_rate_hz,
                    "samples": np.asarray(window.samples, dtype=np.float32),
                }
            )

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic vibration dataset.")
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args(argv)

    with args.params.open(encoding="utf-8") as handle:
        params: dict[str, Any] = yaml.safe_load(handle)["dataset"]

    dataset = build_dataset(params)
    output = Path(params["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output, index=False)
    print(f"Wrote {len(dataset)} windows to {output}")


if __name__ == "__main__":
    main()
