from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import os

def load_layer_timecourses(layer_dir: str | os.PathLike) -> dict[str, np.ndarray]:
    """
    Load Schaefer-400 laminar timecourses for a single subject/run.

    Expected files
    --------------
    Layer_run1_1.npy
    Layer_run1_2.npy
    Layer_run1_3.npy

    Returns
    -------
    layer_data : dict
        Dictionary with keys:
        - "deep"
        - "middle"
        - "superficial"

        Each value has shape (n_parcels, n_timepoints).
    """
    layer_dir = Path(layer_dir)

    file_map = {
        "deep": layer_dir / "Layer_run1_1.npy",
        "middle": layer_dir / "Layer_run1_2.npy",
        "superficial": layer_dir / "Layer_run1_3.npy",
    }

    layer_data = {}
    for layer_name, file_path in file_map.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Could not find expected file: {file_path}")
        arr = np.load(file_path)
        if arr.ndim != 2:
            raise ValueError(
                f"Expected a 2D array for {file_path.name}, got shape {arr.shape}"
            )
        layer_data[layer_name] = arr

    return layer_data


def plot_parcel_timecourse(
    layer_data: dict[str, np.ndarray],
    parcel_idx: int,
    layer_name: str,
    tr: float = 2.0,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the timecourse for one parcel in one cortical layer.

    Parameters
    ----------
    layer_data : dict
        Output of load_layer_timecourses().
    parcel_idx : int
        Parcel index in Schaefer space.
    layer_name : str
        One of: "deep", "middle", "superficial".
    tr : float
        Repetition time in seconds.
    ax : matplotlib Axes, optional
        Existing axis to plot on.

    Returns
    -------
    ax : matplotlib Axes
        Axis containing the plot.
    """
    if layer_name not in layer_data:
        raise ValueError(
            f"Unknown layer '{layer_name}'. Expected one of {list(layer_data.keys())}."
        )

    data = layer_data[layer_name]
    n_parcels, n_timepoints = data.shape

    if not (0 <= parcel_idx < n_parcels):
        raise IndexError(
            f"parcel_idx={parcel_idx} is out of range for {n_parcels} parcels."
        )

    ts = data[parcel_idx, :]
    time_axis = np.arange(n_timepoints) * tr

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5))

    ax.plot(time_axis, ts, linewidth=1.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal")
    ax.set_title(f"Parcel {parcel_idx} | {layer_name} layer")
    ax.grid(alpha=0.25)

    return ax

def compute_adjacency_matrix(
    layer_data: dict[str, np.ndarray],
    layer_name: str,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """
    Compute a parcel-by-parcel adjacency matrix using Pearson correlation
    for one selected layer.

    Parameters
    ----------
    layer_data : dict
        Output of load_layer_timecourses().
    layer_name : str
        One of: "deep", "middle", "superficial".
    zero_diagonal : bool
        If True, set the diagonal to zero for display.

    Returns
    -------
    adj : ndarray, shape (n_parcels, n_parcels)
        Correlation-based adjacency matrix.
    """
    if layer_name not in layer_data:
        raise ValueError(
            f"Unknown layer '{layer_name}'. Expected one of {list(layer_data.keys())}."
        )

    X = layer_data[layer_name]
    adj = np.corrcoef(X)
    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)

    if zero_diagonal:
        np.fill_diagonal(adj, 0.0)

    return adj

def plot_adjacency_matrix(
    adj: np.ndarray,
    layer_name: str,
    ax: Optional[plt.Axes] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> plt.Axes:
    """
    Plot an adjacency matrix as a heatmap.

    Parameters
    ----------
    adj : ndarray
        Parcel-by-parcel adjacency matrix.
    layer_name : str
        Layer name for the title.
    ax : matplotlib Axes, optional
        Existing axis to plot on.
    vmin, vmax : float
        Color limits.

    Returns
    -------
    ax : matplotlib Axes
        Axis containing the plot.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be a square 2D matrix.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(adj, cmap="coolwarm", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(f"Adjacency matrix | {layer_name} layer")
    ax.set_xlabel("Parcel")
    ax.set_ylabel("Parcel")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax

def compute_all_layer_adjacencies(
    layer_data: dict[str, np.ndarray],
    zero_diagonal: bool = True,
) -> dict[str, np.ndarray]:
    """
    Compute adjacency matrices for deep, middle, and superficial layers.

    Returns
    -------
    adj_dict : dict
        Dictionary mapping layer name to adjacency matrix.
    """
    return {
        layer_name: compute_adjacency_matrix(
            layer_data,
            layer_name=layer_name,
            zero_diagonal=zero_diagonal,
        )
        for layer_name in ["deep", "middle", "superficial"]
    }

def build_multiplex_adjacency_from_layers(
    layer_data: dict[str, np.ndarray],
    interlayer_weight: float = 1.0,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """
    Build a multiplex adjacency matrix from the three layer-specific
    adjacency matrices.

    The block diagonal contains the within-layer adjacency matrices,
    and off-diagonal blocks connect the same parcel across layers.

    Parameters
    ----------
    layer_data : dict
        Output of load_layer_timecourses().
    interlayer_weight : float
        Weight of coupling between the same parcel across layers.
    zero_diagonal : bool
        If True, diagonal entries are set to zero.

    Returns
    -------
    M : ndarray, shape (n_parcels * 3, n_parcels * 3)
        Multiplex adjacency matrix.
    """
    layer_order = ["deep", "middle", "superficial"]
    adj_mats = [compute_adjacency_matrix(layer_data, name, zero_diagonal=False) for name in layer_order]

    n_parcels = adj_mats[0].shape[0]
    n_layers = len(adj_mats)

    M = np.zeros((n_parcels * n_layers, n_parcels * n_layers), dtype=float)

    for i, adj in enumerate(adj_mats):
        start = i * n_parcels
        stop = (i + 1) * n_parcels
        M[start:stop, start:stop] = adj

    coupling = np.ones((n_layers, n_layers), dtype=float) - np.eye(n_layers, dtype=float)
    M += interlayer_weight * np.kron(coupling, np.eye(n_parcels, dtype=float))

    if zero_diagonal:
        np.fill_diagonal(M, 0.0)

    return M

def plot_multiplex_adjacency(
    M: np.ndarray,
    ax: Optional[plt.Axes] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> plt.Axes:
    """
    Plot a multiplex adjacency matrix as a heatmap.
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square 2D matrix.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(M, cmap="coolwarm", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title("Multiplex adjacency matrix")
    ax.set_xlabel("Node")
    ax.set_ylabel("Node")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax




def compute_full_laminar_adjacency(
    layer_data: dict[str, np.ndarray],
    layer_order: Optional[list[str]] = None,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """
    Compute a full adjacency matrix across all parcels and all layers.

    The three layer-wise timecourse matrices are concatenated along the parcel axis,
    then a parcel-by-parcel correlation matrix is computed across all layer-specific nodes.

    For Schaefer-400 and 3 layers, this gives a 1200 x 1200 matrix.

    Parameters
    ----------
    layer_data : dict
        Output of load_layer_timecourses().
    layer_order : list of str, optional
        Order in which to concatenate layers.
        Default: ["deep", "middle", "superficial"].
    zero_diagonal : bool
        If True, set the diagonal to zero.

    Returns
    -------
    adj_full : ndarray, shape (n_parcels * n_layers, n_parcels * n_layers)
        Full correlation-based adjacency matrix.
    """
    if layer_order is None:
        layer_order = ["deep", "middle", "superficial"]

    missing = [name for name in layer_order if name not in layer_data]
    if missing:
        raise ValueError(f"Missing layers in layer_data: {missing}")

    arrays = [layer_data[name] for name in layer_order]

    n_parcels = arrays[0].shape[0]
    n_timepoints = arrays[0].shape[1]

    for arr, name in zip(arrays, layer_order):
        if arr.ndim != 2:
            raise ValueError(f"Layer '{name}' is not 2D. Got shape {arr.shape}")
        if arr.shape[0] != n_parcels:
            raise ValueError("All layers must have the same number of parcels.")
        if arr.shape[1] != n_timepoints:
            raise ValueError("All layers must have the same number of time points.")

    all_series = np.concatenate(arrays, axis=0)  # shape: (n_parcels * n_layers, T)

    adj_full = np.corrcoef(all_series)
    adj_full = np.nan_to_num(adj_full, nan=0.0, posinf=0.0, neginf=0.0)

    if zero_diagonal:
        np.fill_diagonal(adj_full, 0.0)

    return adj_full


def plot_full_laminar_adjacency(
    adj_full: np.ndarray,
    n_parcels: int = 400,
    layer_names: Optional[list[str]] = None,
    ax: Optional[plt.Axes] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    show_boundaries: bool = True,
) -> plt.Axes:
    """
    Plot the full laminar adjacency matrix.

    Parameters
    ----------
    adj_full : ndarray
        Full adjacency matrix, typically 1200 x 1200.
    n_parcels : int
        Number of parcels per layer.
    layer_names : list of str, optional
        Names of the layers in the concatenation order.
    ax : matplotlib Axes, optional
        Existing axis to draw on.
    vmin, vmax : float
        Color scale limits.
    show_boundaries : bool
        If True, draw lines separating the layers.

    Returns
    -------
    ax : matplotlib Axes
        Axis containing the plot.
    """
    if adj_full.ndim != 2 or adj_full.shape[0] != adj_full.shape[1]:
        raise ValueError("adj_full must be a square 2D matrix.")

    if layer_names is None:
        layer_names = ["deep", "middle", "superficial"]

    n_layers = len(layer_names)
    expected_size = n_parcels * n_layers
    if adj_full.shape[0] != expected_size:
        raise ValueError(
            f"Expected matrix shape ({expected_size}, {expected_size}), "
            f"got {adj_full.shape}"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(
        adj_full,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    if show_boundaries:
        for k in range(1, n_layers):
            boundary = k * n_parcels - 0.5
            ax.axhline(boundary, color="black", linewidth=1)
            ax.axvline(boundary, color="black", linewidth=1)

    tick_positions = [n_parcels * i + n_parcels / 2 - 0.5 for i in range(n_layers)]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.set_yticklabels(layer_names)

    ax.set_title("Full laminar adjacency matrix")
    ax.set_xlabel("Layer-specific parcels")
    ax.set_ylabel("Layer-specific parcels")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax