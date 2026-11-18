# Noether Link Audit

A telemetry audit for optical ground stations. It uses power, tracking and lock logs to estimate loss mechanisms and compare proposed interventions under a declared receiver model. The model includes turbulence, tip-tilt residual, vibration, static coupling loss, cloud, scintillation, ephemeris bias, keyhole and acquisition.

> **Simulation only.** The sample below runs the audit on synthetic telemetry from a reduced-order model of a hypothetical station. No physical bench, live optical link or on-sky performance is represented.

The reported gigabits are a **power-threshold proxy**: locked time above a fixed sensitivity multiplied by a fixed data rate. They are not modem-measured goodput. Shapley values allocate the model's estimated loss; they do not establish physical causes from telemetry alone. In particular, unexplained coupling loss is assigned to the higher-order turbulence term by assumption. Estimated intervention gains need a controlled test before they can support a recovered-data claim.

## Sample report

[results/link-audit-sample-v1/REPORT.md](results/link-audit-sample-v1/REPORT.md) covers twelve simulated LEO passes over five nights at a hypothetical 0.4 m station, with six kinds of planted fault. The audit read only the logs and the station's declared configuration. The simulator's ground truth was compared afterwards:

- The largest modeled loss category matched the simulator's label on 12 of 12 passes.
- It recovered the planted ephemeris offsets within 3 ms and found the 47 Hz vibration line.
- For five interventions together, the estimated increase in the data proxy was 2,465 Gb; the simulator's counterfactual was 2,453 Gb.
- The report lists where the audit is weakest: it overstates the refocus gain, understates scintillation, and depends on the assumed turbulence outer scale.

The simulator and audit share the coupling model. This checks estimation within that model, not causal identification on a real receiver. The linked v1 report and metrics retain their original bytes and wording; interpret their “delivered,” “recovered” and “upper bound” labels as model outputs under the qualifications above. New reports make these distinctions explicit.

## Run

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

The repository is a uv workspace. Link Audit depends on the independently
installable [Noether Physics package](packages/noether_physics/README.md), which
supports Python 3.11 and 3.12. The complete workspace retains the exact numerical
versions used for the reference results; installing the physics package alone
does not install the audit or its other dependencies.

```sh
make setup
make test
make link-audit-sample
```

`make link-audit-sample` regenerates the synthetic telemetry, the audit, the report and its figures in under a minute on a laptop. Rebuilding the committed coupling tables from scratch (delete `coupling_tables.json` first) adds about a minute. The telemetry is about 200 MB under `results/runs/` and is not committed. [metrics.json](results/link-audit-sample-v1/metrics.json) records every number in the report, with SHA-256 hashes of the sources, the configuration and each telemetry file.

The [Gaussian receiver objective](docs/fiber-coupling.md#differentiating-optical-path)
provides optional JAX gradients of coupling with respect to optical path, at
fixed pupil transmission and receiver geometry. It supports composing actuator
maps with an optical objective; analytic and independent numerical checks are
included. It is a modelling component, not a trained receiver controller.

## Use it on your station

The audit starts with exported logs and a receiver configuration. The [receiver pilot protocol](docs/receiver-pilot.md) distinguishes observations, model estimates and the controlled measurements needed to evaluate a change. Write to founders@noether.one to discuss a representative sample and whether its signals can answer your question.

## Layout

| Path | Contents |
|---|---|
| `projects/link-audit/` | Audit, simulator, attribution and report code |
| `packages/noether_physics/` | Independent `noether-physics` distribution: Gaussian overlap and optional derivatives |
| `configs/link-audit-sample.yaml` | Sample scenario: station, link budget, nights and planted causes |
| `docs/fiber-coupling.md` | Coupling method and its analytic checks |
| `results/` | Sample report, metrics and the coupling comparison |
| `tests/` | Unit, analytic and smoke tests |

## License

Apache-2.0. Built by [Noether Labs](https://www.noether.one).
