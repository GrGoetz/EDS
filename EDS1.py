from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import hyperspy.api as hs
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths


BIN_WIDTH_KEV = 0.02
GA_LA_KEV = 1.09799
GA_KA_KEV = 9.24312
ASLS_SMOOTHNESS = 1e6
ASLS_ASYMMETRY = 0.0005
ASLS_MAX_ITER = 20
BACKGROUND_OFFSET = -75.0
ASLS_EDGE_PAD = 10
BACKGROUND_SMOOTH_SIGMA = 0.5
BACKGROUND_TAIL_START_KEV = 13.0
BACKGROUND_TAIL_FRACTION = 0.76
BACKGROUND_TAIL_BLEND_WIDTH_KEV = 3.0
MIN_SUBTRACTED_COUNT = 1e-3
LOW_ENERGY_PEAK_MAX_KEV = 12.0
LOW_ENERGY_MIN_REL_HEIGHT = 0.0012
LOW_ENERGY_MIN_REL_PROMINENCE = 0.00045
LOW_ENERGY_MIN_DISTANCE_POINTS = 4
MID_ENERGY_PEAK_MAX_KEV = 16.5
MID_ENERGY_MIN_REL_HEIGHT = 0.0015
MID_ENERGY_MIN_REL_PROMINENCE = 0.0007
MID_ENERGY_MIN_DISTANCE_POINTS = 8
HIGH_ENERGY_MIN_REL_HEIGHT = 0.00022
HIGH_ENERGY_MIN_REL_PROMINENCE = 0.0002
HIGH_ENERGY_MIN_DISTANCE_POINTS = 20
HIGH_ENERGY_SMOOTH_SIGMA = 2.0
HIGH_ENERGY_MIN_WIDTH_POINTS = 3
HIGH_ENERGY_MAX_PEAKS = 3
Y_SCALE = "linear"

def energy_step(energy):
    return float(np.median(np.diff(energy))) if len(energy) > 1 else BIN_WIDTH_KEV


def resolve_y_scale(y_scale=None):
    return Y_SCALE if y_scale is None else y_scale


def apply_y_axis(ax, y_scale, *series):
    y_scale = resolve_y_scale(y_scale)
    positive_values = []
    for values in series:
        array = np.asarray(values, dtype=float)
        positive_values.extend(array[array > 0])

    if not positive_values:
        return

    ymin = min(positive_values)
    ymax = max(positive_values)

    if y_scale == "log":
        ax.set_yscale("symlog", linthresh=max(1.0, ymin), linscale=1.0)
        ax.set_ylim(bottom=ymin * 0.8, top=ymax * 1.2)
    else:
        ax.set_yscale("linear")
        ax.set_ylim(bottom=0, top=ymax * 1.05)


def add_peak_label_headroom(ax, peak_fits, factor=6.0):
    if not peak_fits:
        return

    _, current_top = ax.get_ylim()
    peak_top = max(float(peak["height"]) for peak in peak_fits)
    ax.set_ylim(top=max(current_top, peak_top * float(factor)))


def create_signal_from_counts(filepath, energy, counts, bin_width_kev):
    signal = hs.signals.Signal1D(np.asarray(counts, dtype=float))
    signal.metadata.General.title = Path(filepath).stem
    signal.metadata.Signal.quantity = "Counts"

    axis = signal.axes_manager.signal_axes[0]
    axis.name = "Energy"
    axis.units = "keV"
    axis.scale = energy[1] - energy[0] if len(energy) > 1 else bin_width_kev
    axis.offset = energy[0]
    return signal


def calibrated_energy_axis(n_points, offset, scale):
    return offset + scale * np.arange(n_points, dtype=float)


def peak_channel_position(peak, axis_offset, axis_scale):
    return float((peak["center"] - axis_offset) / axis_scale)


def closest_peak_within_tolerance(peak_fits, reference_energy, tolerance, excluded=None):
    excluded = excluded or set()
    candidates = [
        peak for peak in peak_fits
        if id(peak) not in excluded and abs(peak["center"] - reference_energy) <= tolerance
    ]
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda peak: (abs(peak["center"] - reference_energy), -peak["height"]),
    )


def choose_reference_peaks(peak_fits, reference_energies):
    if len(peak_fits) < len(reference_energies):
        raise ValueError(
            f"Need at least {len(reference_energies)} fitted peaks for calibration, "
            f"but only found {len(peak_fits)}."
        )

    if len(reference_energies) != 2:
        raise ValueError("Current calibration expects exactly two reference energies.")

    low_ref, high_ref = [float(value) for value in reference_energies]
    low_peak = closest_peak_within_tolerance(peak_fits, low_ref, tolerance=0.35)
    excluded = {id(low_peak)} if low_peak is not None else set()
    high_peak = closest_peak_within_tolerance(peak_fits, high_ref, tolerance=0.75, excluded=excluded)

    if low_peak is not None and high_peak is not None and high_peak["center"] > low_peak["center"]:
        return [low_peak, high_peak]

    low_peak = closest_peak_within_tolerance(peak_fits, low_ref, tolerance=0.75)
    excluded = {id(low_peak)} if low_peak is not None else set()
    high_peak = closest_peak_within_tolerance(peak_fits, high_ref, tolerance=1.5, excluded=excluded)

    if low_peak is not None and high_peak is not None and high_peak["center"] > low_peak["center"]:
        return [low_peak, high_peak]

    raise ValueError(
        "Could not identify Ga Lα and Ga Kα near their expected energies. "
        "Check whether those peaks are present and clearly detected."
    )


def calibrate_signal_with_reference_peaks(signal, peak_fits, reference_energies, reference_labels):
    reference_peaks = choose_reference_peaks(peak_fits, reference_energies)
    axis = signal.axes_manager.signal_axes[0]
    axis_offset = float(axis.offset)
    axis_scale = float(axis.scale)
    reference_positions = [
        peak_channel_position(peak, axis_offset, axis_scale) for peak in reference_peaks
    ]

    if abs(reference_positions[1] - reference_positions[0]) <= 1e-12:
        raise ValueError("Calibration peaks collapsed onto the same channel position.")

    scale = (
        float(reference_energies[1] - reference_energies[0])
        / float(reference_positions[1] - reference_positions[0])
    )
    offset = float(reference_energies[0] - scale * reference_positions[0])

    calibrated_signal = signal.deepcopy()
    calibrated_axis = calibrated_signal.axes_manager.signal_axes[0]
    calibrated_axis.scale = scale
    calibrated_axis.offset = offset

    calibration = {
        "offset": offset,
        "scale": scale,
        "reference_labels": list(reference_labels),
        "reference_energies": [float(value) for value in reference_energies],
        "reference_peaks": reference_peaks,
        "reference_positions": reference_positions,
    }
    return calibrated_signal, calibration


def plot_spectrum_comparison(energy, counts, signal, subtract_bg, background, bg_method, y_scale=None):
    y_scale = resolve_y_scale(y_scale)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(energy, counts, label="With Bremsstrahlung", color="black", linestyle=":")

    processed_label = "Original Spectrum"
    if subtract_bg and background is not None:
        processed_label = f"Background Subtracted ({bg_method})"
        ax.plot(
            energy,
            background,
            label=f"Estimated Background ({bg_method})",
            color="tab:blue",
            linestyle=":",
            linewidth=1.5,
        )

    ax.plot(
        signal.axes_manager.signal_axes[0].axis,
        signal.data,
        label=processed_label,
        color="tab:orange",
    )
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    if subtract_bg and background is not None:
        apply_y_axis(ax, y_scale, counts, background, signal.data)
    else:
        apply_y_axis(ax, y_scale, counts, signal.data)
    ax.set_title("Spectrum Comparison")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    plt.show()


def read_eds_txt(filepath):
    filepath = Path(filepath)

    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    energy = []
    counts = []
    in_spectrum = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.upper().startswith("#SPECTRUM"):
            in_spectrum = True
            continue

        if stripped.upper().startswith("#ENDOFDATA"):
            break

        if not in_spectrum:
            continue

        parts = [x.strip() for x in stripped.split(",")]

        if len(parts) < 2:
            continue

        try:
            e = float(parts[0])
            c = float(parts[1])
        except ValueError:
            continue

        energy.append(e)
        counts.append(c)

    if not energy:
        raise ValueError(f"No EDS data found in file: {filepath}")

    return energy, counts


def make_signal_from_data(data, axis_template, title=None, quantity=None):
    signal = hs.signals.Signal1D(np.asarray(data, dtype=float))
    if title is not None:
        signal.metadata.General.title = title
    if quantity is not None:
        signal.metadata.Signal.quantity = quantity

    axis = signal.axes_manager.signal_axes[0]
    axis.name = axis_template.name
    axis.units = axis_template.units
    axis.scale = axis_template.scale
    axis.offset = axis_template.offset

    return signal


def estimate_polynomial_background(energy, counts, degree=2):
    n_windows = min(80, max(20, len(counts) // 25))
    edges = np.linspace(0, len(counts), n_windows + 1, dtype=int)

    anchor_x = []
    anchor_y = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop - start < 2:
            continue

        window_energy = energy[start:stop]
        window_counts = counts[start:stop]
        min_index = int(np.argmin(window_counts))
        anchor_x.append(window_energy[min_index])
        anchor_y.append(window_counts[min_index])

    if len(anchor_x) <= degree:
        raise ValueError(
            "Not enough points to estimate polynomial background. "
            "Try a lower --bg-degree."
        )

    coeffs = np.polyfit(anchor_x, anchor_y, degree)
    return np.polyval(coeffs, energy)


def estimate_asls_background(
    counts,
    smoothness=ASLS_SMOOTHNESS,
    asymmetry=ASLS_ASYMMETRY,
    max_iter=ASLS_MAX_ITER,
):
    counts = np.asarray(counts, dtype=float)
    n_points = len(counts)

    if n_points < 3:
        return counts.copy()

    pad = max(0, min(int(ASLS_EDGE_PAD), max(0, n_points - 1)))
    padded_counts = np.pad(counts, pad_width=pad, mode="reflect") if pad else counts
    n_padded = len(padded_counts)

    diff_matrix = np.diff(np.eye(n_padded), 2, axis=0)
    penalty = smoothness * (diff_matrix.T @ diff_matrix)
    weights = np.ones(n_padded, dtype=float)

    for _ in range(max_iter):
        weighted_system = np.diag(weights) + penalty
        baseline = np.linalg.solve(weighted_system, weights * padded_counts)
        new_weights = np.where(padded_counts > baseline, asymmetry, 1.0 - asymmetry)

        if np.allclose(new_weights, weights):
            break

        weights = new_weights

    if pad:
        baseline = baseline[pad:-pad]

    baseline = gaussian_filter1d(baseline, sigma=BACKGROUND_SMOOTH_SIGMA, mode="nearest")
    return baseline


def stabilize_background_tail(energy, counts, background):
    energy = np.asarray(energy, dtype=float)
    counts = np.asarray(counts, dtype=float)
    background = np.asarray(background, dtype=float)

    if len(background) == 0:
        return background

    transition_center = float(BACKGROUND_TAIL_START_KEV)
    blend_width = max(float(BACKGROUND_TAIL_BLEND_WIDTH_KEV), 1e-6)
    transition = np.clip((energy - transition_center) / blend_width, 0.0, 1.0)

    if not np.any(transition > 0):
        return background

    tail_counts = gaussian_filter1d(counts, sigma=8.0, mode="nearest")
    tail_floor = np.maximum(
        1.0,
        float(BACKGROUND_TAIL_FRACTION) * tail_counts,
    )
    lifted_background = np.maximum(background, tail_floor)
    return (1.0 - transition) * background + transition * lifted_background


def find_signal_peaks(
    signal,
    max_peaks=50,
    min_rel_height=0.0007,
    min_rel_prominence=0.00025,
    min_distance_points=1,
):
    energy = np.asarray(signal.axes_manager.signal_axes[0].axis, dtype=float)
    counts = np.asarray(signal.data, dtype=float)

    if len(counts) < 3 or np.max(counts) <= 0:
        return []

    max_count = float(np.max(counts))
    low_mask = energy < float(LOW_ENERGY_PEAK_MAX_KEV)
    mid_mask = (energy >= float(LOW_ENERGY_PEAK_MAX_KEV)) & (energy < float(MID_ENERGY_PEAK_MAX_KEV))
    high_mask = energy >= float(MID_ENERGY_PEAK_MAX_KEV)

    peak_indices = []
    peak_heights = []

    if np.any(low_mask):
        low_indices, low_properties = find_peaks(
            counts[low_mask],
            height=max_count * max(float(min_rel_height), float(LOW_ENERGY_MIN_REL_HEIGHT)),
            prominence=max_count * max(float(min_rel_prominence), float(LOW_ENERGY_MIN_REL_PROMINENCE)),
            distance=max(max(1, int(min_distance_points)), int(LOW_ENERGY_MIN_DISTANCE_POINTS)),
        )
        low_offset = int(np.flatnonzero(low_mask)[0])
        peak_indices.extend((low_indices + low_offset).tolist())
        peak_heights.extend(low_properties.get("peak_heights", []).tolist())

    if np.any(mid_mask):
        mid_indices, mid_properties = find_peaks(
            counts[mid_mask],
            height=max_count * max(float(min_rel_height), float(MID_ENERGY_MIN_REL_HEIGHT)),
            prominence=max_count * max(float(min_rel_prominence), float(MID_ENERGY_MIN_REL_PROMINENCE)),
            distance=max(max(1, int(min_distance_points)), int(MID_ENERGY_MIN_DISTANCE_POINTS)),
        )
        mid_offset = int(np.flatnonzero(mid_mask)[0])
        peak_indices.extend((mid_indices + mid_offset).tolist())
        peak_heights.extend(mid_properties.get("peak_heights", []).tolist())

    if np.any(high_mask):
        high_counts = gaussian_filter1d(
            counts[high_mask],
            sigma=float(HIGH_ENERGY_SMOOTH_SIGMA),
            mode="nearest",
        )
        high_indices, high_properties = find_peaks(
            high_counts,
            height=max_count * min(float(min_rel_height), float(HIGH_ENERGY_MIN_REL_HEIGHT)),
            prominence=max_count * min(float(min_rel_prominence), float(HIGH_ENERGY_MIN_REL_PROMINENCE)),
            distance=max(max(1, int(min_distance_points)), int(HIGH_ENERGY_MIN_DISTANCE_POINTS)),
            width=int(HIGH_ENERGY_MIN_WIDTH_POINTS),
        )
        if len(high_indices) > int(HIGH_ENERGY_MAX_PEAKS):
            ranking = np.argsort(high_properties.get("peak_heights", np.array([])))[::-1][: int(HIGH_ENERGY_MAX_PEAKS)]
            high_indices = high_indices[ranking]
            for key, values in high_properties.items():
                high_properties[key] = values[ranking]
        high_offset = int(np.flatnonzero(high_mask)[0])
        peak_indices.extend((high_indices + high_offset).tolist())
        peak_heights.extend(high_counts[high_indices].tolist())

    if len(peak_indices) == 0:
        return []

    peak_indices = np.asarray(peak_indices, dtype=int)
    peak_heights = np.asarray(peak_heights, dtype=float)

    widths = peak_widths(counts, peak_indices, rel_height=0.5)[0]

    peaks = sorted(
        (
            {
                "center": float(energy[index]),
                "height": float(peak_heights[i]),
                "width": float(widths[i] * energy_step(energy)),
            }
            for i, index in enumerate(peak_indices)
        ),
        key=lambda peak: peak["height"],
        reverse=True,
    )[:max_peaks]

    return sorted(peaks, key=lambda peak: peak["center"])


def annotate_peaks(ax, peak_fits, y_scale=None, show_markers=True, show_lines=True, color="tab:red"):
    y_scale = resolve_y_scale(y_scale)
    _, y_top = ax.get_ylim()
    label_factors = [1.18, 1.32, 1.48]

    for label_index, peak in enumerate(peak_fits):
        center = peak["center"]
        peak_height = float(peak["height"])
        label_y = min(peak_height * label_factors[label_index % len(label_factors)], y_top * 0.78)

        if show_lines:
            ax.axvline(center, color=color, linestyle="--", linewidth=0.8, alpha=0.35, zorder=1)
        if show_markers:
            ax.scatter([center], [peak_height], color=color, s=24, zorder=5)
        if y_scale == "linear":
            ax.annotate(
                f"{center:.3f} keV",
                xy=(center, peak_height),
                xytext=(0, 8 + 10 * (label_index % 3)),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=90,
                color=color,
                fontsize=8,
                clip_on=True,
            )
        else:
            ax.annotate(
                f"{center:.3f} keV",
                xy=(center, peak_height),
                xytext=(center, label_y),
                textcoords="data",
                ha="center",
                va="bottom",
                rotation=90,
                color=color,
                fontsize=8,
                clip_on=False,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8, "alpha": 0.6},
            )


def strongest_signal_index(signals):
    if not signals:
        raise ValueError("Need at least one signal to choose the strongest spectrum.")

    return int(
        np.argmax([float(np.max(np.asarray(signal.data, dtype=float))) for signal in signals])
    )


def plot_overlay_with_peak_labels(signals, y_scale=None):
    y_scale = resolve_y_scale(y_scale)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    plotted_counts = []

    for index, signal in enumerate(signals):
        energy = np.asarray(signal.axes_manager.signal_axes[0].axis, dtype=float)
        counts = np.asarray(signal.data, dtype=float)
        plotted_counts.append(counts)
        title = getattr(signal.metadata.General, "title", f"Spectrum {index + 1}")
        color = colors[index % len(colors)] if colors else None
        ax.plot(energy, counts, label=title, color=color)

    reference_index = strongest_signal_index(signals)
    reference_signal = signals[reference_index]
    reference_peak_fits = getattr(reference_signal, "_peak_fits", [])
    reference_color = colors[reference_index % len(colors)] if colors else "tab:orange"
    apply_y_axis(ax, y_scale, *plotted_counts)
    if y_scale == "log":
        add_peak_label_headroom(ax, reference_peak_fits)
    annotate_peaks(
        ax,
        reference_peak_fits,
        y_scale=y_scale,
        show_markers=False,
        color=reference_color,
    )

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title("Processed Spectra with Peak Labels", pad=28)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend()
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    plt.show()


def plot_peak_identification(signal, peak_fits, calibration=None, y_scale=None):
    y_scale = resolve_y_scale(y_scale)
    fig, ax = plt.subplots(figsize=(11, 7))
    energy = np.asarray(signal.axes_manager.signal_axes[0].axis, dtype=float)
    counts = np.asarray(signal.data, dtype=float)

    ax.plot(energy, counts, color="tab:orange", label="Processed Spectrum")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    apply_y_axis(ax, y_scale, counts)
    if y_scale == "log":
        add_peak_label_headroom(ax, peak_fits)
    annotate_peaks(ax, peak_fits, y_scale=y_scale, show_markers=True)
    title = f"Peak Identification: {signal.metadata.General.title}"
    if calibration is not None:
        title = f"{title} (Ga calibrated)"
        labels = calibration["reference_labels"]
        energies = calibration["reference_energies"]
        positions = calibration["reference_positions"]
        ax.text(
            0.98,
            0.98,
            (
                f"{labels[0]} -> {energies[0]:.5f} keV at channel {positions[0]:.2f}\n"
                f"{labels[1]} -> {energies[1]:.5f} keV at channel {positions[1]:.2f}\n"
                f"offset = {calibration['offset']:.5f} keV, bin width = {calibration['scale']:.6f} keV"
            ),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )
    ax.set_title(title, pad=28)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend()
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    plt.show()


def build_eds_signal(
    filepath,
    bin_width_kev=BIN_WIDTH_KEV,
    calibrate=True,
    known_peaks=(GA_LA_KEV, GA_KA_KEV),
    expected_peaks=("Ga Lα", "Ga Kα"),
    subtract_bg=True,
    bg_method="asls",
    bg_degree=2,
    fit_peaks=True,
    max_peaks=50,
    min_rel_height=0.0007,
    min_rel_prominence=0.00025,
    min_distance_points=1,
    y_scale=None,
):
    y_scale = resolve_y_scale(y_scale)
    energy, counts = read_eds_txt(filepath)
    signal = create_signal_from_counts(filepath, energy, counts, bin_width_kev)

    background = None

    if subtract_bg:
        signal, background = subtract_background(signal, method=bg_method, degree=bg_degree)

    calibration = None
    if calibrate and fit_peaks:
        preliminary_peak_fits = find_signal_peaks(
            signal,
            max_peaks=max_peaks,
            min_rel_height=min_rel_height,
            min_rel_prominence=min_rel_prominence,
            min_distance_points=min_distance_points,
        )
        try:
            signal, calibration = calibrate_signal_with_reference_peaks(
                signal,
                preliminary_peak_fits,
                known_peaks,
                expected_peaks,
            )
            energy = calibrated_energy_axis(len(energy), calibration["offset"], calibration["scale"])
        except ValueError as exc:
            print(f"Skipping Ga calibration for {Path(filepath).name}: {exc}")

    plot_spectrum_comparison(energy, counts, signal, subtract_bg, background, bg_method, y_scale)

    if fit_peaks:
        peak_fits = find_signal_peaks(
            signal,
            max_peaks=max_peaks,
            min_rel_height=min_rel_height,
            min_rel_prominence=min_rel_prominence,
            min_distance_points=min_distance_points,
        )
        signal._peak_fits = peak_fits
        signal._calibration = calibration
        plot_peak_identification(signal, peak_fits, calibration=calibration, y_scale=y_scale)
    else:
        signal._peak_fits = []
        signal._calibration = calibration

    return signal


def subtract_background(signal, method="asls", degree=2):
    """
    Subtract background from the EDS signal.

    Parameters:
        signal (hs.signals.Signal1D): The EDS signal to process.
        method (str): The background subtraction method (default: "asls").
        degree (int): Degree of the polynomial for background fitting (if applicable).

    Returns:
        tuple[hs.signals.Signal1D, np.ndarray]: The corrected signal and estimated background.
    """
    axis = signal.axes_manager.signal_axes[0]
    energy = np.asarray(axis.axis, dtype=float)
    counts = np.asarray(signal.data, dtype=float)

    if method == "polynomial":
        background = estimate_polynomial_background(energy, counts, degree=degree)
    elif method == "asls":
        background = estimate_asls_background(counts)
    elif method == "snip":
        raise NotImplementedError(
            "SNIP background subtraction is not available for this generic Signal1D "
            "in the installed HyperSpy version. Use --bg-method asls or polynomial."
        )
    else:
        raise ValueError(f"Unsupported background subtraction method: {method}")

    background = np.asarray(background, dtype=float) + float(BACKGROUND_OFFSET)
    background = stabilize_background_tail(energy, counts, background)
    background = np.clip(background, 0, counts)
    corrected = np.clip(counts - background, float(MIN_SUBTRACTED_COUNT), None)

    signal = make_signal_from_data(
        corrected,
        axis,
        getattr(signal.metadata.General, "title", None),
        getattr(signal.metadata.Signal, "quantity", None),
    )

    return signal, background


def plot_eds(
    files,
    overlay=True,
    bin_width_kev=BIN_WIDTH_KEV,
    calibrate=True,
    subtract_bg=True,
    bg_method="asls",
    bg_degree=2,
    fit_peaks=True,
    max_peaks=50,
    min_rel_height=0.0007,
    min_rel_prominence=0.00025,
    min_distance_points=1,
    y_scale=None,
):
    """Plot one or multiple EDS spectra with HyperSpy."""
    y_scale = resolve_y_scale(y_scale)
    signals = [
        build_eds_signal(
            file,
            bin_width_kev=bin_width_kev,
            calibrate=calibrate,
            subtract_bg=subtract_bg,
            bg_method=bg_method,
            bg_degree=bg_degree,
            fit_peaks=fit_peaks,
            max_peaks=max_peaks,
            min_rel_height=min_rel_height,
            min_rel_prominence=min_rel_prominence,
            min_distance_points=min_distance_points,
            y_scale=y_scale,
        )
        for file in files
    ]

    if overlay:
        plot_overlay_with_peak_labels(signals, y_scale=y_scale)
    else:
        for signal in signals:
            signal.plot()

    return signals


def main():
    global Y_SCALE
    parser = argparse.ArgumentParser(
        description="Read EDS spectra from txt files, process them, and plot with HyperSpy."
    )
    parser.add_argument("files", nargs="+", help="One or more EDS txt files")
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Plot each spectrum in a separate figure instead of overlaying them"
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=BIN_WIDTH_KEV,
        help=f"Fallback bin width in keV if only one data point is present (default: {BIN_WIDTH_KEV})"
    )
    parser.add_argument(
        "--calibrate-ga",
        dest="calibrate",
        action="store_true",
        default=True,
        help="Calibrate the energy axis from the fitted Ga Lα and Ga Kα peaks (default: enabled)"
    )
    parser.add_argument(
        "--no-calibrate-ga",
        dest="calibrate",
        action="store_false",
        help="Disable the Ga-based energy calibration"
    )
    parser.add_argument(
        "--subtract-bg",
        dest="subtract_bg",
        action="store_true",
        default=True,
        help="Subtract background from the spectra (default: enabled)"
    )
    parser.add_argument(
        "--no-subtract-bg",
        dest="subtract_bg",
        action="store_false",
        help="Disable background subtraction"
    )
    parser.add_argument(
        "--bg-method",
        type=str,
        default="asls",
        choices=["asls", "polynomial", "snip"],
        help="Background subtraction method (default: asls)"
    )
    parser.add_argument(
        "--bg-degree",
        type=int,
        default=2,
        help="Degree of the polynomial for background fitting (default: 2)"
    )
    parser.add_argument(
        "--fit-peaks",
        dest="fit_peaks",
        action="store_true",
        default=True,
        help="Detect peaks with HyperSpy and annotate their energies (default: enabled)"
    )
    parser.add_argument(
        "--no-fit-peaks",
        dest="fit_peaks",
        action="store_false",
        help="Disable peak detection"
    )
    parser.add_argument(
        "--max-peaks",
        type=int,
        default=50,
        help="Maximum number of peaks to annotate (default: 50)"
    )
    parser.add_argument(
        "--min-rel-height",
        type=float,
        default=0.0007,
        help="Minimum peak height relative to the spectrum maximum (default: 0.0007)"
    )
    parser.add_argument(
        "--min-rel-prominence",
        type=float,
        default=0.00025,
        help="Minimum relative slope threshold for HyperSpy peak detection (default: 0.00025)"
    )
    parser.add_argument(
        "--min-distance-points",
        type=int,
        default=1,
        help="Minimum spacing between detected peaks in data points (default: 1)"
    )
    parser.add_argument(
        "--y-scale",
        type=str,
        default="linear",
        choices=["linear", "log"],
        help="Y-axis scaling for plots (default: linear)"
    )
    args = parser.parse_args()
    Y_SCALE = args.y_scale

    plot_eds(
        args.files,
        overlay=not args.separate,
        bin_width_kev=args.bin_width,
        calibrate=args.calibrate,
        subtract_bg=args.subtract_bg,
        bg_method=args.bg_method,
        bg_degree=args.bg_degree,
        fit_peaks=args.fit_peaks,
        max_peaks=args.max_peaks,
        min_rel_height=args.min_rel_height,
        min_rel_prominence=args.min_rel_prominence,
        min_distance_points=args.min_distance_points,
    )


if __name__ == "__main__":
    main()
