# Calibration-aware control for an optical downlink receiver

**Prospective evaluation protocol · 18 November 2026.** This document defines
the measurements and decisions needed for a receiver pilot. It is not a
completed experiment, an agreed partner specification, or a hardware result.

The question is whether identifying changes in a receiver's optical response
allows predictive or learned control to preserve useful received power under
delay, pointing disturbance, and calibration drift. The comparison must include
a tuned classical controller and a strong classical predictor. A learned
controller earns its place by improving the receiver outcome within the same
latency, actuator, and calibration budgets.

## Start with one receiver

Specify the optical path before selecting an algorithm: wavelength and bandwidth,
aperture and obscuration, receiver mode or detector geometry, polarization,
sensor, steering mirror, deformable mirror if present, and the existing control
law. Record the operating elevation range and whether the data come from a
bench, a horizontal atmospheric link, or a satellite pass. Those settings are
not interchangeable.

For an ideal single-mode receiver, the optical objective is the normalized
overlap between the incident field and the receiving mode:

\[
\eta = \frac{|\langle E,u\rangle|^2}
{\langle E,E\rangle\langle u,u\rangle}.
\]

The [Gaussian overlap component](fiber-coupling.md) specifies the normalization
and checks analytic limits and independent focal-plane calculations. A measured
receiver may need a different mode, polarization model, or detector objective.
The overlap component provides a numerical reference and an optional
differentiable objective for real optical path at fixed pupil and geometry.
It is not an end-to-end receiver model or a model of an identified instrument.

Received fiber power also depends on incident flux and optical transmission.
Phase correction cannot reconstruct light blocked by clouds or guarantee
recovery from amplitude scintillation. Track those effects separately. A
quantum receiver additionally needs its own count, background, error-rate, and
protocol criteria; classical modem goodput is not an appropriate substitute.

## Minimum recording request

Begin with a short, contiguous recording and a calibration session to validate
the interface. The number of independent passes or nights for an outcome study
is determined subsequently from development variability and a declared minimum
useful effect, before opening evaluation data.

| Record | Required meaning |
| --- | --- |
| Time | Exposure start/end, observation-available time, command issue/apply time, clock domains and synchronization uncertainty; sequence numbers and dropped-frame flags. |
| Sensor | Raw wavefront-sensor data or calibrated slopes; for a quad cell, all channels, their sum, gain, background and saturation flags. A centroid alone cannot identify high-order phase. |
| Actuation | Requested and applied FSM/DM commands, units, sign conventions, transfer functions, delays, stroke/slew limits and saturation; controller state and settings. |
| Calibration | Response matrices or calibration exposures, their timestamps, pupil mapping, sensor reference and calibration identifiers. Record any changes during a pass. |
| Receiver | Power before and after fiber injection where available, detector calibration/noise floor, optical transmission and receiving-mode definition. Separate coupling from changing incident flux. |
| Link outcome | Lock/acquisition/recovery state; actual modem delivered-bit/error counters and rate/FEC configuration when available. Document resets and counter wrap. |
| Conditions | Pass identity, elevation and tracking reference; available wind/turbulence and weather observations, with units and timestamps. |

Missing channels remain missing. They are not replaced by simulator truth.
Logs stay with the instrument team unless a separate sharing agreement permits
transfer. An offline analysis needs no credentials, remote control, or command
access. Publication of partner data or identifying results is a separate decision.

## Evidence stages

```text
recording + calibration
        |
        v
timing / units / missing-data qualification ----> insufficient evidence
        |
        v
development fit -> separate-pass prediction check -> model rejected or qualified
        |
        v
shadow candidate commands (existing controller remains in charge)
        |
        v
partner-approved, controlled bench comparison -> measured outcome / no benefit
        |
        v
separate field protocol and authorization
```

1. **Recording qualification.** Use [Noether Replay](https://github.com/noetherlabs/noether-replay)
   for applicable timing, calibration, and command-provenance checks. Report
   unavailable or ambiguous mappings. A consistent recording is not evidence
   that a different controller would have worked.
2. **System identification.** Fit optical response and actuator dynamics on
   development data. Test measurement predictions on separate passes or nights.
   Compare a fixed nominal model, an identified model, and a simpler empirical
   model. Check identifiability: changes in gain, illumination and alignment can
   explain the same measurement. Use approved calibration probes where needed.
3. **Shadow comparison.** Produce time-stamped candidate commands without
   applying them. Measure computation time, freshness, deadline misses, limits
   and abstentions. Recorded trajectories reflect the incumbent controller;
   replaying them does not establish counterfactual closed-loop coupling.
4. **Controlled bench comparison.** After an instrument-specific review and
   authorization, use repeatable disturbance realizations or randomized,
   appropriately blocked controller trials. Measure actual receiver outcomes
   and the full sensor-to-actuator delay. Retain failed and interrupted trials.
5. **Field confirmation.** Use a new protocol for atmospheric links or satellite
   passes. A horizontal-link result does not by itself establish performance on
   a slant-path satellite downlink.

## Comparison and acceptance

Separate three hypotheses rather than attributing every improvement to learning:

- **Identification:** does updating the receiver model improve held-out optical
  prediction and control relative to keeping the nominal calibration fixed?
  Apply both models to the same controller before changing the policy.
- **Measurement choice:** does an adaptive calibration probe recover more useful
  receiver operation than it consumes? Include its exposure, photon, interruption
  and actuation costs over a predeclared horizon; compare a fixed schedule with
  the same budget.
- **Qualification:** can an observable model-discrepancy score predict when a
  candidate loses its advantage? Fit the rule on development data, freeze it,
  and measure missed degradations, unnecessary withdrawals and actual switching
  cost on independent trials. A low prediction error alone is not a stability
  certificate.

Freeze the following before evaluation: configurations, splits, seeds where
applicable, controller selection, thresholds, calibration/probe budgets,
minimum useful gain, allowed regression, latency limits, and stopping rules.
All numerical acceptance values depend on the actual receiver and are
**unassigned in this prospective document**. No pilot is qualified by leaving
them unspecified.

Compare the incumbent, a retuned integrator or equivalent classical loop, a
classical predictive controller such as an identified AR/LQG model, and the
candidate. Include the same identified model without learned control to isolate
whether any gain comes from calibration or the policy. Apply identical sensing,
delay, actuator and training-data restrictions. Separately test nominal
conditions, drift, low signal, pointing/vibration, saturation and missing data;
report regressions as well as improvements.

For a fiber-coupled receiver, retain the absolute coupling distribution,
predeclared lower-tail statistic, fraction and duration of power fades, and
reacquisition time. Normalize power only when calibrated incident-power data
support it. Report observation coverage and censored/dropped intervals.

If actual delivered bits are available, compare them per pass or predefined
window. A power-threshold model instead computes

\[
\widehat B = \sum_t R_t\,\mathbf 1[\text{locked}_t]
\,\mathbf 1[P_t\ge P_{\min,t}]\,\Delta t_t.
\]

This is a model-derived proxy: it omits decoder behavior, errors, overhead,
retransmissions and unmodeled rate adaptation. Label it separately from a modem
counter. A model allocation such as a Shapley decomposition partitions its
assumed loss function; it does not prove physical causation or recoverable bits.

Use independent disturbance realizations, passes or nights as the analysis
units. Adjacent high-rate frames are correlated and must not masquerade as
independent trials. Prespecify paired contrasts and multiplicity handling,
retain uncertainty and all failure outcomes, and report both absolute and
relative effects with their denominators. No retuning follows evaluation.

Fallback is itself an intervention. Measure transfer transients, incumbent
state reconstruction, saturation and time to recovery. The existing controller
can remain the operational default without being assumed safe under every
fault. A monitor requires its own false-alarm, miss and detection-delay study
before it can authorize switching.

## Existing evidence and remaining research

| Component | Public evidence | Remaining receiver question |
| --- | --- | --- |
| [Noether AO / Prediction Boundary](https://www.noether.one/research/prediction-boundary) | 600 held-out simulated trials; prediction improves wavefront error in 5 of 12 conditions, 7 uncertain; learned mixture does not beat both references. | Does the effect survive receiver-specific sensing, spatial coupling and measured latency? |
| [Noether Twin](https://github.com/noetherlabs/noether-twin) | Development study of optical registration, model resolution and held-out measurement prediction. | Which parameters can be identified from the receiver's available channels? |
| [ProbeBench](https://github.com/noetherlabs/probebench) | Fixed/adaptive calibration comparisons under finite photon and exposure budgets. | Is the information gained by probing worth lost receiver time or power? |
| [Noether Replay](https://github.com/noetherlabs/noether-replay) | Offline telemetry reconstruction and evidence checks. | Can the required timing and applied-command history be established on this instrument? |
| [Fiber overlap](../results/fiber-coupling-v1/README.md) | 13 deterministic cases, three grids, analytic and independent Fourier checks. | Does the receiving-mode model predict measured coupled power? |
| [Recovery study](https://github.com/noetherlabs/noether-ao/tree/main/results/recovery_study_v1) | Scheduled simulated transfers with retained harmful and uncertain outcomes. | Can an instrument-specific transition avoid unacceptable coupling loss? |

Predictive and learned optical-link control are established research directions,
not novelty claims of this programme. Robert et al.'s
[2026 laboratory study](https://doi.org/10.1364/OE.603458) evaluates VAR-LQG
control for LEO-link adaptive optics; Parvizi et al.'s
[2023 study](https://doi.org/10.3390/photonics10121371) provides a reinforcement
learning environment for sensorless downlink correction. These motivate stronger
baselines than an untuned integrator.

The proposed contribution is a testable connection between receiver
identification, measurement choice, control performance and the decision to
retain or withdraw a candidate under calibration change. Establishing it needs
new receiver-specific evidence. None of the existing software checks or
simulation reports establishes a deployed link improvement.
