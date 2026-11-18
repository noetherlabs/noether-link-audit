# Noether Link Audit

A telemetry loss audit for optical ground stations, with a synthetic estimation check. Simulation only: the sample uses logs from a hypothetical station, not a physical bench, live link or on-sky data.

The audit reads power, tracking, FSM command, pointing, offload and lock logs plus the declared station configuration. It estimates a power-threshold data proxy and allocates model-conditional losses with Shapley values. This proxy is not measured modem goodput; the allocation does not identify physical causes without additional evidence. Unexplained coupling residual is modeled as higher-order turbulence. Intervention estimates hold other model terms fixed and require controlled validation. Planted simulator labels provide a check within the assumed model.

- [Sample report](../../results/link-audit-sample-v1/REPORT.md): twelve passes, five nights, planted faults, audit findings and the validation against truth.
- [Scenario](../../configs/link-audit-sample.yaml): station, link budget, nights and planted causes.

## Run

```sh
make link-audit-sample
.venv/bin/python -m pytest tests/test_link_audit.py
```

Bulk telemetry (about 200 MB) is written to `results/runs/link-audit-sample-v1/` and is not committed; its SHA-256 hashes are recorded in `metrics.json`.

## Modules

| Module | Role |
|---|---|
| `station.py` | Station configuration, pass geometry, Hufnagel–Valley atmosphere with tracking pseudo-wind, tip-tilt loop transfer functions |
| `coupling_tables.py` | Monte Carlo single-mode coupling tables from HCIPy screens and the qualified `noether_physics.coupling` overlap |
| `simulate.py` | Synthetic pass telemetry with planted causes; ground truth kept in memory |
| `audit.py` | Telemetry-only estimation: extinction, scintillation, noise, residual variance, r0, vibration, static loss, timing bias, keyhole, acquisition |
| `attribution.py` | Threshold data proxy and model-conditional Shapley allocation |
| `pipeline.py`, `report.py` | End-to-end sample, metrics, figures and report |

The audit never imports the simulator's truth. It reads the written parquet logs back from disk. The simulator and the audit share the coupling model, so the validation tests estimation within that model, not independent coupling physics or causal identification. The retained v1 report has its original wording; its delivered/recovered gigabits are threshold proxies and its ideal-AO “upper bound” is an optimistic model scenario, not a receiver-independent bound. See the [receiver pilot protocol](../../docs/receiver-pilot.md) for the evidence needed on an instrument.
