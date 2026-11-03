"""Deterministic spatial sensitivity check of the Gaussian fiber overlap.

Run from the repository root and redirect stdout to a new JSON file:
    .venv/bin/python scripts/compare_fiber_coupling.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from noether_physics.coupling import gaussian_fiber_coupling

WAVELENGTH_M = 1.55e-6
FOCAL_LENGTH_M = 0.1
PUPIL_RADIUS_M = 0.001
# Ideal centered Gaussian's circular-pupil optimum: exp(q) = 1 + 2*q.
Q_OPTIMUM = 1.2564312086261697
MODE_RADIUS_M = WAVELENGTH_M * FOCAL_LENGTH_M * np.sqrt(Q_OPTIMUM) / (
    np.pi * PUPIL_RADIUS_M
)
RESOLUTIONS = (64, 128, 256)


def compare() -> dict:
    rows = []
    cases = [(name, rms, 0.0, 1.0) for name in ("tilt_x", "defocus", "astigmatism")
             for rms in (0.3, 0.6, 1.0)]
    cases.extend(("none", 0.0, obscuration, scale) for obscuration, scale in (
        (0.0, 1.0), (0.3, 1.0), (0.0, 0.7), (0.0, 1.4)))
    for name, target_rms, obscuration, mode_scale in cases:
        samples = []
        for n in RESOLUTIONS:
            pitch = 2.4 * PUPIL_RADIUS_M / n
            axis = (np.arange(n) - (n - 1) / 2) * pitch / PUPIL_RADIUS_M
            x, y = np.meshgrid(axis, axis)
            radius2 = x**2 + y**2
            mask = (radius2 <= 1) & (radius2 >= obscuration**2)
            amplitude = mask.astype(float)
            phase = {"none": np.zeros_like(x), "tilt_x": x,
                     "defocus": 2 * radius2 - 1, "astigmatism": x**2 - y**2}[name]
            phase -= np.mean(phase[mask])
            if target_rms:
                phase *= target_rms / np.sqrt(np.mean(phase[mask]**2))
            opd = phase * (WAVELENGTH_M / (2 * np.pi))
            kwargs = dict(pixel_pitch_m=pitch, wavelength_m=WAVELENGTH_M,
                          focal_length_m=FOCAL_LENGTH_M, mode_radius_m=MODE_RADIUS_M * mode_scale)
            eta = gaussian_fiber_coupling(amplitude, opd, **kwargs)
            eta0 = gaussian_fiber_coupling(amplitude, np.zeros_like(opd), **kwargs)
            # Spatial analogues, not reconstruction of the plant's undocumented
            # modal-to-pupil basis. Ratios remove the zero-aberration throughput.
            sigma2 = float(np.mean(phase[mask]**2))
            tilt2 = sigma2 if name == "tilt_x" else 0.0
            samples.append({
                "grid_size": n,
                "overlap_efficiency": eta,
                "zero_aberration_efficiency": eta0,
                "relative_overlap": eta / eta0,
                "spatial_phase_rms_rad": np.sqrt(sigma2),
                "variance_only_relative_proxy": np.exp(-sigma2),
                "variance_plus_tilt_relative_proxy": np.exp(-sigma2 - 0.35 * tilt2),
            })
        rows.append({
            "phase_profile": name,
            "target_phase_rms_rad": target_rms,
            "central_obscuration_radius_fraction": obscuration,
            "focal_mode_radius_scale": mode_scale,
            "samples": samples,
            "last_refinement_absolute_efficiency_change": abs(
                samples[-1]["overlap_efficiency"] - samples[-2]["overlap_efficiency"]),
        })
    root = Path(__file__).resolve().parents[1]
    source_paths = ("packages/noether_physics/src/noether_physics/coupling.py",
                    "scripts/compare_fiber_coupling.py")
    return {
        "schema": "gaussian_fiber_overlap_comparison_v1",
        "scope": "deterministic_scalar_spatial_sensitivity_not_controller_evaluation",
        "normalization": "incident_pupil_power_times_full_plane_gaussian_mode_power",
        "sampling": "piecewise_constant_pupil_field_exact_gaussian_cell_integrals",
        "proxy_mapping": "spatial_variance_analogue_not_a_reanalysis_of_modal_plant_results",
        "parameters": {
            "wavelength_m": WAVELENGTH_M,
            "focal_length_m": FOCAL_LENGTH_M,
            "pupil_radius_m": PUPIL_RADIUS_M,
            "focal_mode_radius_m": float(MODE_RADIUS_M),
            "grid_width_over_pupil_radius": 2.4,
            "grid_sizes": RESOLUTIONS,
        },
        "software": {"numpy": np.__version__, "scipy": scipy.__version__},
        "source_sha256": {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                          for path in source_paths},
        "cases": rows,
    }


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2, sort_keys=True, allow_nan=False))
