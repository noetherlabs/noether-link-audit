# Noether Link Audit

A telemetry-only loss audit for optical ground stations, with a validated sample report. Simulation only: the sample uses synthetic logs from a hypothetical station, not a physical bench, live link or on-sky data.

The audit reads what a station logs and its declared configuration, then attributes lost data to causes with Shapley values. The logs are fiber and tracking-sensor power, tracking residuals, FSM commands, pointing, offloads and lock state. The causes are turbulence, tip-tilt residual, vibration, static coupling loss, cloud, scintillation, ephemeris bias, keyhole and acquisition. The audit also estimates what each fix would recover. The simulator that produces the sample logs keeps every planted cause as ground truth, so the audit's answers can be checked.

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
| `attribution.py` | Counterfactual data accounting and Shapley attribution |
| `pipeline.py`, `report.py` | End-to-end sample, metrics, figures and report |

The audit never imports the simulator's truth. It reads the written parquet logs back from disk. The simulator and the audit share the coupling model, so the validation tests estimation from telemetry, not the coupling physics.
