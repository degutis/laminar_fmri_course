
"""
Utility functions for Exercise 1 of the laminar fMRI course.

This module focuses on:
1. Quality-control visualization of T1/functional alignment
2. Overlay plots for layers and parcels on top of the anatomical image
3. Re-generating layers with LAYNII from a rim file

The functions are intentionally simple and readable for teaching purposes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Tuple
import io
from IPython.display import Image, display
import imageio.v2 as imageio

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap


def load_nifti(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    return data, img.affine


def _normalize_slice(slice2d: np.ndarray, robust: bool = True) -> np.ndarray:
    arr = np.asarray(slice2d, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if robust:
        lo, hi = np.percentile(arr, [1, 99])
    else:
        lo, hi = arr.min(), arr.max()
    if hi <= lo:
        return np.zeros_like(arr, dtype=float)
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo)


def _get_slice(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError("Expected a 3D NIfTI image.")
    slicer = [slice(None)] * 3
    slicer[axis] = index
    arr = data[tuple(slicer)]

    arr = data[tuple(slicer)]
    arr = np.rot90(arr)
    arr = np.flipud(arr)

    return arr


def _default_slice_indices(data: np.ndarray, axis: int, n_slices: int = 24) -> np.ndarray:
    size = data.shape[axis]
    start = max(1, int(round(size * 0.15)))
    stop = min(size - 2, int(round(size * 0.85)))
    return np.linspace(start, stop, n_slices).astype(int)


def make_alignment_gif(
    t1_path: str | os.PathLike,
    func_mean_path: str | os.PathLike,
    output_gif: Optional[str | os.PathLike] = None,
    axis: int = 2,
    frame_duration: float = 1.5,
    n_slices: int = 24,
    show: bool = True,
):
    t1, _ = load_nifti(t1_path)
    func, _ = load_nifti(func_mean_path)

    if t1.shape != func.shape:
        raise ValueError(
            f"T1 and func mean shapes do not match: {t1.shape} vs {func.shape}. "
            "They should already be in the same space for this exercise."
        )

    indices = _default_slice_indices(t1, axis=axis, n_slices=n_slices)
    frames = []

    for idx in indices:
        t1_slice = _normalize_slice(_get_slice(t1, axis, idx))
        func_slice = _normalize_slice(_get_slice(func, axis, idx))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(t1_slice, cmap="gray", origin="lower")
        ax.set_title(f"T1 in func space | slice {idx}")
        ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.renderer.buffer_rgba())[:, :, :3])
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(func_slice, cmap="gray", origin="lower")
        ax.set_title(f"Functional mean | slice {idx}")
        ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.renderer.buffer_rgba())[:, :, :3])
        plt.close(fig)

    if output_gif is not None:
        output_gif = Path(output_gif)
        output_gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_gif, frames, duration=frame_duration, loop=0)
        if show:
            display(Image(filename=str(output_gif)))
        return output_gif

    buffer = io.BytesIO()
    imageio.mimsave(buffer, frames, format="GIF", duration=frame_duration, loop=0)
    gif_bytes = buffer.getvalue()

    if show:
        display(Image(data=gif_bytes))

    return gif_bytes


def plot_overlay_on_t1(
    t1_path: str | os.PathLike,
    overlay_path: str | os.PathLike,
    output_png: Optional[str | os.PathLike] = None,
    axis: int = 2,
    slices: Optional[Sequence[int]] = None,
    ncols: int = 4,
    alpha: float = 0.45,
    cmap_name: str = "viridis",
    title: Optional[str] = None,
    threshold: float = 0.0,
    show: bool = True,
):
    t1, _ = load_nifti(t1_path)
    overlay, _ = load_nifti(overlay_path)

    if t1.shape != overlay.shape:
        raise ValueError(f"T1 and overlay shapes do not match: {t1.shape} vs {overlay.shape}.")

    if slices is None:
        slices = _default_slice_indices(t1, axis=axis, n_slices=8)

    slices = list(slices)
    nrows = int(np.ceil(len(slices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, slices):
        t1_slice = _normalize_slice(_get_slice(t1, axis, idx))
        overlay_slice = _get_slice(overlay, axis, idx).astype(float)
        masked = np.ma.masked_where(np.abs(overlay_slice) <= threshold, overlay_slice)

        ax.imshow(t1_slice, cmap="gray", origin="lower")
        ax.imshow(masked, cmap=cmap_name, alpha=alpha, origin="lower", interpolation="nearest")
        ax.set_title(f"slice {idx}")
        ax.axis("off")

    for ax in axes[len(slices):]:
        ax.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=14, y=0.98)

    fig.tight_layout()

    if output_png is not None:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def plot_discrete_overlay_on_t1(
    t1_path: str | os.PathLike,
    overlay_path: str | os.PathLike,
    output_png: Optional[str | os.PathLike] = None,
    axis: int = 2,
    slices: Optional[Sequence[int]] = None,
    labels: Optional[Sequence[int]] = None,
    colors: Optional[Sequence[str]] = None,
    ncols: int = 4,
    alpha: float = 0.45,
    title: Optional[str] = None,
    show: bool = True,
):
    t1, _ = load_nifti(t1_path)
    overlay, _ = load_nifti(overlay_path)

    if t1.shape != overlay.shape:
        raise ValueError(f"T1 and overlay shapes do not match: {t1.shape} vs {overlay.shape}.")

    if slices is None:
        slices = _default_slice_indices(t1, axis=axis, n_slices=8)

    if labels is None:
        labels = sorted([int(v) for v in np.unique(overlay) if v != 0])

    if colors is None:
        base = ["#440154", "#21918c", "#fde725", "#fdae61", "#d7191c", "#2b83ba"]
        colors = base[: len(labels)]

    cmap = ListedColormap(colors)
    label_to_index = {lab: i for i, lab in enumerate(labels)}

    slices = list(slices)
    nrows = int(np.ceil(len(slices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, slices):
        t1_slice = _normalize_slice(_get_slice(t1, axis, idx))
        overlay_slice = _get_slice(overlay, axis, idx)

        mapped = np.full_like(overlay_slice, np.nan, dtype=float)
        for lab in labels:
            mapped[overlay_slice == lab] = label_to_index[lab]

        masked = np.ma.masked_invalid(mapped)

        ax.imshow(t1_slice, cmap="gray", origin="lower")
        ax.imshow(
            masked,
            cmap=cmap,
            alpha=alpha,
            origin="lower",
            interpolation="nearest",
            vmin=0,
            vmax=max(len(labels) - 1, 1),
        )
        ax.set_title(f"slice {idx}")
        ax.axis("off")

    for ax in axes[len(slices):]:
        ax.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=14, y=0.98)

    fig.tight_layout()

    if output_png is not None:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def run_laynii_layers(
    rim_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    n_layers: int = 3,
    equivol: bool = True,
    thickness: bool = True,
    laynii_executable: str = "LN2_LAYERS",
    output_stem: str = "layers_from_rim",
) -> Path:
    rim_path = Path(rim_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which(laynii_executable) is None:
        raise FileNotFoundError(
            f"Could not find '{laynii_executable}'. Install LAYNII and make sure "
            "LN2_LAYERS is on your PATH."
        )

    output_path = output_dir / f"{output_stem}.nii.gz"
    cmd = [
        laynii_executable,
        "-rim", str(rim_path),
        "-nr_layers", str(n_layers),
        "-output", str(output_path),
    ]
    if equivol:
        cmd.append("-equivol")
    if thickness:
        cmd.append("-thickness")

    print("Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_path


def simple_rgb_overlay(
    background: np.ndarray,
    mask: np.ndarray,
    color: Tuple[float, float, float] = (1.0, 0.2, 0.2),
    alpha: float = 0.45,
) -> np.ndarray:
    bg = _normalize_slice(background)
    rgb = np.dstack([bg, bg, bg])
    mask = mask.astype(bool)
    for c in range(3):
        rgb[..., c] = np.where(mask, (1 - alpha) * rgb[..., c] + alpha * color[c], rgb[..., c])
    return np.clip(rgb, 0, 1)
