
"""Utility functions for Exercise 2 of a laminar fMRI course.

This module keeps the analysis intentionally lightweight and readable:
- extract High and Low load time courses for 3 cortical layers
- plot group time courses with SEM
- plot layer-wise load effects as violin plots with subject-level connections
- run a paired t-test on the load difference between two layers

Expected directory structure
----------------------------
/data/S01/func/Load_long_TENTzero_response_condition_High_prcchg.nii
/data/S01/func/Load_long_TENTzero_response_condition_Low_prcchg.nii
/data/S01/anat/dlpfc_l_parcel_map.nii
/data/S01/anat/cop_l_parcel_map.nii
/data/S01/anat/cop_r_parcel_map.nii
/data/S01/anat/fpn_r_parcel_map.nii
/data/S01/anat/ds_scaled_rim_layers_equidist_3layers.nii
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

DEFAULT_PARTICIPANTS = [f"S{i:02d}" for i in range(1, 10)]

ROI_FILES = {
    "dlpfc_left": "dlpfc_l_parcel_map.nii",
    "dlpfc_l": "dlpfc_l_parcel_map.nii",
    "dlpfc": "dlpfc_l_parcel_map.nii",
    "cop_left": "cop_l_parcel_map.nii",
    "cop_l": "cop_l_parcel_map.nii",
    "cop": "cop_l_parcel_map.nii",
    "cop_right": "cop_r_parcel_map.nii",
    "cop_r": "cop_r_parcel_map.nii",
    "dlpfc_right": "fpn_r_parcel_map.nii",
    "fpn_r": "fpn_r_parcel_map.nii",
}

LAYER_FILE = "ds_scaled_rim_layers_equidist_3layers.nii"
LAYER_CODES = [1, 2, 3]
LAYER_NAMES = {1: "Deep", 2: "Middle", 3: "Superficial"}


def resolve_roi(roi: str = "dlpfc_left") -> str:
    """Return the ROI mask filename for a user-friendly ROI name."""
    key = roi.lower()
    if key not in ROI_FILES:
        valid = ", ".join(sorted(set(ROI_FILES)))
        raise ValueError(f"Unknown ROI '{roi}'. Valid choices include: {valid}")
    return ROI_FILES[key]


def _safe_mean_across_voxels(masked_2d: np.ndarray) -> np.ndarray:
    """Mean across voxels while ignoring zeros, infs, and nans."""
    arr = masked_2d.astype(float)
    arr[arr == 0] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return np.nanmean(arr, axis=0)


def extract_subject_timecourses(
    data_dir: str | Path,
    subject: str,
    roi: str = "dlpfc_left",
    add_tentzero_padding: bool = True,
) -> dict:
    """Extract High and Low load time courses for one subject and one ROI.

    Returns
    -------
    dict with keys:
        subject, roi, high, low, n_voxels
    high / low shape = (3, n_timepoints)
    n_voxels shape = (3,)
    """
    data_dir = Path(data_dir)
    subj_dir = data_dir / subject
    func_dir = subj_dir / "func"
    anat_dir = subj_dir / "anat"

    high_file = func_dir / "Load_long_TENTzero_response_condition_High_prcchg.nii"
    low_file = func_dir / "Load_long_TENTzero_response_condition_Low_prcchg.nii"
    roi_file = anat_dir / resolve_roi(roi)
    layer_file = anat_dir / LAYER_FILE

    for f in [high_file, low_file, roi_file, layer_file]:
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}")

    high_img = nib.load(str(high_file)).get_fdata()
    low_img = nib.load(str(low_file)).get_fdata()
    roi_img = nib.load(str(roi_file)).get_fdata()
    layer_img = nib.load(str(layer_file)).get_fdata()

    roi_mask = (roi_img != 0).astype(float)
    roi_layers = roi_mask * layer_img

    high_layers = []
    low_layers = []
    n_voxels = []

    for layer_code in LAYER_CODES:
        layer_mask = roi_layers == layer_code
        n_voxels.append(int(layer_mask.sum()))

        layer_mask_4d = np.repeat(layer_mask[..., None], high_img.shape[-1], axis=3)

        high_masked = (high_img * layer_mask_4d).reshape(-1, high_img.shape[-1])
        low_masked = (low_img * layer_mask_4d).reshape(-1, low_img.shape[-1])

        high_tc = _safe_mean_across_voxels(high_masked)
        low_tc = _safe_mean_across_voxels(low_masked)

        if add_tentzero_padding:
            high_tc = np.concatenate([[0.0], high_tc, [0.0]])
            low_tc = np.concatenate([[0.0], low_tc, [0.0]])

        high_layers.append(high_tc)
        low_layers.append(low_tc)

    return {
        "subject": subject,
        "roi": roi,
        "high": np.vstack(high_layers),
        "low": np.vstack(low_layers),
        "n_voxels": np.asarray(n_voxels),
    }


def load_group_data(
    data_dir: str | Path,
    participants: Sequence[str] = DEFAULT_PARTICIPANTS,
    roi: str = "dlpfc_left",
) -> dict:
    """Load all subjects for one ROI."""
    subjects = [extract_subject_timecourses(data_dir, s, roi=roi) for s in participants]

    high = np.stack([s["high"] for s in subjects], axis=0)  # subjects x layers x time
    low = np.stack([s["low"] for s in subjects], axis=0)
    n_voxels = np.stack([s["n_voxels"] for s in subjects], axis=0)

    return {
        "participants": list(participants),
        "roi": roi,
        "high": high,
        "low": low,
        "difference": high - low,
        "n_voxels": n_voxels,
        "time_seconds": np.arange(high.shape[-1]) * 2.0,
    }


def _sem(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """NaN-safe SEM."""
    n = np.sum(np.isfinite(x), axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=1)
    return sd / np.sqrt(n)


def plot_group_timecourses(
    group_data: dict,
    layers_to_plot: Sequence[int] = (1, 2, 3),
    conditions: Sequence[str] = ("high", "low"),
    figsize: tuple[float, float] = (15, 4.5),
):
    """Plot High and Low time courses for selected layers.

    Notes
    -----
    - layers_to_plot uses layer codes 1, 2, 3
    - one subplot per layer, arranged horizontally
    - all subplots share a global y-axis range within the ROI
    - SEM is shown as a shaded envelope around the mean
    """
    time = group_data["time_seconds"]
    layer_indices = [l - 1 for l in layers_to_plot]

    all_vals = []
    for cond in conditions:
        all_vals.append(group_data[cond][:, layer_indices, :])
    all_vals = np.concatenate(all_vals, axis=1)
    y_min = np.nanmin(all_vals)
    y_max = np.nanmax(all_vals)
    y_pad = 0.08 * (y_max - y_min if y_max > y_min else 1.0)

    fig, axes = plt.subplots(
        1, len(layer_indices), figsize=figsize, sharex=True, sharey=True, constrained_layout=True
    )
    if len(layer_indices) == 1:
        axes = [axes]

    for ax, layer_idx, layer_code in zip(axes, layer_indices, layers_to_plot):
        for cond in conditions:
            data = group_data[cond][:, layer_idx, :]
            mean_tc = np.nanmean(data, axis=0)
            sem_tc = _sem(data, axis=0)
            label = cond.capitalize()
            ax.plot(time, mean_tc, label=label)
            ax.fill_between(time, mean_tc - sem_tc, mean_tc + sem_tc, alpha=0.2)

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(f"{LAYER_NAMES[layer_code]} layer")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    axes[0].set_ylabel("Percent signal change")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Group time courses in {group_data['roi']}", y=1.03)
    return fig, axes


def summarise_window(
    group_data: dict,
    condition: str = "difference",
    time_window: tuple[int, int] = (4, 6),
) -> np.ndarray:
    """Average a time window for each subject and layer.

    Parameters
    ----------
    condition : "high", "low", or "difference"
    time_window : tuple of inclusive indices, e.g. (4, 6)
        These are indices in the padded TENTzero time course.
    """
    start, end = time_window
    data = group_data[condition][:, :, start : end + 1]
    return np.nanmean(data, axis=-1)  # subjects x layers


def plot_load_effect_violin(
    group_data: dict,
    roi: str | None = None,
    time_window: tuple[int, int] = (4, 6),
    layers: Sequence[int] = (1, 2, 3),
    figsize: tuple[float, float] = (8, 5),
    jitter: float = 0.05,
    seed: int = 7,
):
    """Plot High-Low as a violin plot for selected layers.

    Adds lightly transparent subject-wise connecting lines and jittered points.
    """
    rng = np.random.default_rng(seed)
    summary = summarise_window(group_data, condition="difference", time_window=time_window)
    layer_indices = [l - 1 for l in layers]
    values = summary[:, layer_indices]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    positions = np.arange(1, len(layers) + 1)

    violin_data = [values[:, i] for i in range(values.shape[1])]
    vp = ax.violinplot(violin_data, positions=positions, widths=0.8, showmeans=False, showextrema=False)
    for body in vp["bodies"]:
        body.set_alpha(0.25)

    # same color for all subjects
    base_color = "C0"
    x_jittered = np.column_stack(
        [pos + rng.uniform(-jitter, jitter, size=values.shape[0]) for pos in positions]
    )

    for subj in range(values.shape[0]):
        ax.plot(x_jittered[subj], values[subj], color=base_color, alpha=0.25, linewidth=1.2)
        ax.scatter(x_jittered[subj], values[subj], color=base_color, alpha=0.75, s=28)

    layer_labels = [LAYER_NAMES[l] for l in layers]
    ax.set_xticks(positions)
    ax.set_xticklabels(layer_labels)
    ax.axhline(0, linestyle="--", linewidth=1)
    roi_label = roi if roi is not None else group_data["roi"]
    ax.set_title(f"Load effect (High - Low) in {roi_label}\nAveraged across timepoints {time_window[0]}-{time_window[1]}")
    ax.set_ylabel("Percent signal change difference")
    return fig, ax


def paired_ttest_between_layers(
    group_data: dict,
    layer_x: int = 1,
    layer_y: int = 3,
    time_window: tuple[int, int] = (4, 6),
) -> dict:
    """Paired t-test on the load difference between two layers.

    This tests:
        (High - Low in layer X) versus (High - Low in layer Y)

    within one ROI and one time window.
    """
    summary = summarise_window(group_data, condition="difference", time_window=time_window)
    x = summary[:, layer_x - 1]
    y = summary[:, layer_y - 1]
    t_stat, p_value = stats.ttest_rel(x, y, nan_policy="omit")

    return {
        "roi": group_data["roi"],
        "time_window": time_window,
        "layer_x": layer_x,
        "layer_y": layer_y,
        "layer_x_name": LAYER_NAMES[layer_x],
        "layer_y_name": LAYER_NAMES[layer_y],
        "mean_x": float(np.nanmean(x)),
        "mean_y": float(np.nanmean(y)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n_subjects": int(np.sum(np.isfinite(x) & np.isfinite(y))),
        "x_values": x,
        "y_values": y,
    }


def print_ttest_report(result: dict) -> None:
    """Print a compact, student-friendly summary."""
    print(f"ROI: {result['roi']}")
    print(f"Window: timepoints {result['time_window'][0]}-{result['time_window'][1]}")
    print(f"Comparison: High-Low in {result['layer_x_name']} vs {result['layer_y_name']}")
    print(f"n = {result['n_subjects']}")
    print(f"Mean {result['layer_x_name']}: {result['mean_x']:.4f}")
    print(f"Mean {result['layer_y_name']}: {result['mean_y']:.4f}")
    print(f"Paired t-test: t = {result['t_stat']:.3f}, p = {result['p_value']:.4f}")
