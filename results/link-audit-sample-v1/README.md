# Interpreting the retained v1 sample

This directory contains a synthetic estimation study of twelve passes at a hypothetical optical ground station. It contains no measured receiver goodput, physical-bench result or on-sky result.

The original [report](REPORT.md), [metrics](metrics.json), figures and provenance are retained unchanged. Read their terminology as follows:

| Original label | Meaning in this experiment |
| --- | --- |
| Delivered / observed delivered Gb | Locked samples at or above a fixed power sensitivity, multiplied by sample duration and a fixed data rate. This is a threshold proxy, not a modem counter. |
| Lost Gb attributed to a cause | A Shapley allocation of the assumed counterfactual loss model. Exact accounting does not prove that a physical cause is identifiable from the logs. |
| Higher-order turbulence | Unexplained coupling residual assigned to this category by the model. Other aberrations, alignment changes, calibration error or model discrepancy could enter the same residual on a real receiver. |
| Recovered Gb by a fix | Increase in the threshold proxy under a modeled intervention with other terms held fixed. No intervention was applied to an instrument. |
| Ideal AO upper bound | An optimistic ideal-modal-correction scenario using coupling quantiles. It is not a general upper bound on an actual receiver's improvement. |

The 12/12 largest-category agreement and the estimated 2,465 Gb combined gain versus the simulator's 2,453 Gb counterfactual are checks within a shared coupling model. They do not validate causal identification or predict a measured gain on another receiver.

Newly generated metrics include a `measurement_contract` and an additive `threshold_delivered_gb` alias; legacy numeric fields remain available. This interpretation does not recalculate or replace the retained v1 values. The [receiver pilot protocol](../../docs/receiver-pilot.md) defines the real measurements and controlled comparisons needed before claiming receiver improvement.
