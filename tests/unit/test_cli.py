"""Unit tests for the dataset generation CLI."""

from pathlib import Path

import pandas as pd
import yaml

from digital_twin.cli import build_dataset, main
from services.common.schemas import FaultType

PARAMS = {
    "output_path": "unused.parquet",
    "windows_per_class": 3,
    "sampling_rate_hz": 12000,
    "duration_s": 0.25,
    "shaft_speed_hz_min": 25.0,
    "shaft_speed_hz_max": 35.0,
    "severity_min": 0.2,
    "severity_max": 1.0,
    "noise_std": 0.05,
    "seed": 1,
}


def test_build_dataset_has_balanced_classes() -> None:
    dataset = build_dataset(PARAMS)
    assert len(dataset) == 3 * len(FaultType)
    counts = dataset["label"].value_counts()
    assert set(counts) == {3}


def test_build_dataset_is_reproducible() -> None:
    first = build_dataset(PARAMS)
    second = build_dataset(PARAMS)
    assert first["asset_id"].tolist() == second["asset_id"].tolist()
    assert (first["samples"].iloc[5] == second["samples"].iloc[5]).all()


def test_healthy_windows_have_zero_severity() -> None:
    dataset = build_dataset(PARAMS)
    healthy = dataset[dataset["label"] == FaultType.HEALTHY.value]
    assert (healthy["severity"] == 0.0).all()


def test_main_writes_parquet_file(tmp_path: Path) -> None:
    output = tmp_path / "out" / "dataset.parquet"
    params_file = tmp_path / "params.yaml"
    params_file.write_text(
        yaml.safe_dump({"dataset": {**PARAMS, "output_path": str(output)}}),
        encoding="utf-8",
    )
    main(["--params", str(params_file)])
    loaded = pd.read_parquet(output)
    assert len(loaded) == 3 * len(FaultType)
    assert len(loaded["samples"].iloc[0]) == 3000
