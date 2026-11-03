# Noether Link Audit

A telemetry-only link-loss audit for optical ground stations. It reads what a station already logs and attributes every lost gigabit to a cause, then estimates what each fix would recover. The causes it separates are turbulence, tip-tilt residual, vibration, static coupling loss, cloud, scintillation, ephemeris bias, keyhole and acquisition.

> **Simulation only.** The sample below runs the audit on synthetic telemetry from a reduced-order model of a hypothetical station. No physical bench, live optical link or on-sky performance is represented.

## Sample report

[results/link-audit-sample-v1/REPORT.md](results/link-audit-sample-v1/REPORT.md) covers twelve simulated LEO passes over five nights at a hypothetical 0.4 m station, with six kinds of planted fault. The audit read only the logs and the station's declared configuration. The simulator's ground truth was compared afterwards:

- The audit named the right largest cause on 12 of 12 passes.
- It recovered the planted ephemeris offsets within 3 ms and found the 47 Hz vibration line.
- For the five practical fixes together, it estimated 2,465 Gb recovered; the truth was 2,453 Gb.
- The report lists where the audit is weakest: it overstates the refocus gain, understates scintillation, and depends on the assumed turbulence outer scale.

## Run

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
make setup
make test
make link-audit-sample
```

`make link-audit-sample` regenerates the synthetic telemetry, the audit, the report and its figures in under a minute on a laptop. Rebuilding the committed coupling tables from scratch (delete `coupling_tables.json` first) adds about a minute. The telemetry is about 200 MB under `results/runs/` and is not committed. [metrics.json](results/link-audit-sample-v1/metrics.json) records every number in the report, with SHA-256 hashes of the sources, the configuration and each telemetry file.

## Use it on your station

The audit needs logs, not access to your systems. The end of the report lists the data it uses. Write to founders@noether.one for a free audit of 10 to 20 passes.

## Layout

| Path | Contents |
|---|---|
| `projects/link-audit/` | Audit, simulator, attribution and report code |
| `packages/noether_physics/` | Qualified scalar overlap with a Gaussian fiber mode |
| `configs/link-audit-sample.yaml` | Sample scenario: station, link budget, nights and planted causes |
| `docs/fiber-coupling.md` | Coupling method and its analytic checks |
| `results/` | Sample report, metrics and the coupling comparison |
| `tests/` | Unit, analytic and smoke tests |

## License

Apache-2.0. Built by [Noether Labs](https://www.noether.one).
