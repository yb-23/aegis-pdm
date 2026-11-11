# AEGIS-PdM

[![CI](https://github.com/yb-23/aegis-pdm/actions/workflows/ci.yml/badge.svg)](https://github.com/yb-23/aegis-pdm/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/cost-0%E2%82%AC-brightgreen)

**Anomaly Edge-to-cloud Inference & Governance System for Predictive Maintenance.**

A TinyML INT8 model on STM32 detects vibration anomalies in real time, a cloud-native backend fuses vibration and synthetic thermography to estimate Remaining Useful Life, and an MLOps pipeline (DVC, MLflow, GitHub Actions, Hugging Face Hub) manages training, governance and drift-triggered redeployment. Infrastructure cost: 0 EUR.

## Status

Sprint 0 (foundations) in progress.

## Development setup

```powershell
uv sync
uv run pre-commit install
uv run pytest
```
