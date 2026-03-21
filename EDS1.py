from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import hyperspy.api as hs
from scipy.signal import find_peaks, peak_widths


BIN_WIDTH_KEV = 0.02
GA_LA_KEV = 1.09799
GA_KA_KEV = 9.24312

def energy_step(energy):
    return float(np.median(np.diff(energy))) if len(energy) > 1 else BIN_WIDTH_KEV


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


def plot_spectrum_comparison(energy, counts, signal, subtract_bg, background, bg_method):
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
    ax.set_title("Spectrum Comparison")
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


def estimate_asls_background(counts, smoothness=1e6, asymmetry=0.01, max_iter=20):
    counts = np.asarray(counts, dtype=float)
    n_points = len(counts)

    if n_points < 3:
        return counts.copy()

    diff_matrix = np.diff(np.eye(n_points), 2, axis=0)
    penalty = smoothness * (diff_matrix.T @ diff_matrix)
    weights = np.ones(n_points, dtype=float)

    for _ in range(max_iter):
        weighted_system = np.diag(weights) + penalty
        baseline = np.linalg.solve(weighted_system, weights * counts)
        new_weights = np.where(counts > baseline, asymmetry, 1.0 - asymmetry)

        if np.allclose(new_weights, weights):
            break

        weights = new_weights

    return baseline


def find_signal_peaks(
    signal,
    max_peaks=20,
    min_rel_height=0.001,
    min_rel_prominence=0.0003,
    min_distance_points=1,
):
    energy = np.asarray(signal.axes_manager.signal_axes[0].axis, dtype=float)
    counts = np.asarray(signal.data, dtype=float)

    if len(counts) < 3 or np.max(counts) <= 0:
        return []

    peak_indices, properties = find_peaks(
        counts,
        height=float(np.max(counts) * min_rel_height),
        prominence=float(np.max(counts) * min_rel_prominence),
        distance=max(1, int(min_distance_points)),
    )

    if len(peak_indices) == 0:
        return []

    widths = peak_widths(counts, peak_indices, rel_height=0.5)[0]

    peaks = sorted(
        (
            {
                "center": float(energy[index]),
                "height": float(properties["peak_heights"][i]),
                "width": float(widths[i] * energy_step(energy)),
            }
            for i, index in enumerate(peak_indices)
        ),
        key=lambda peak: peak["height"],
        reverse=True,
    )[:max_peaks]

    return sorted(peaks, key=lambda peak: peak["center"])


def annotate_peaks(ax, peak_fits, show_markers=True, color="tab:red"):
    for label_index, peak in enumerate(peak_fits):
        center = peak["center"]
        if show_markers:
            ax.scatter([center], [peak["height"]], color=color, s=24, zorder=5)
        ax.annotate(
            f"{center:.3f} keV",
            xy=(center, peak["height"]),
            xytext=(0, 10 + 12 * (label_index % 3)),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            color=color,
            fontsize=9,
        )


def strongest_signal_index(signals):
    if not signals:
        raise ValueError("Need at least one signal to choose the strongest spectrum.")

    return int(
        np.argmax([float(np.max(np.asarray(signal.data, dtype=float))) for signal in signals])
    )


def plot_overlay_with_peak_labels(signals):
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for index, signal in enumerate(signals):
        energy = np.asarray(signal.axes_manager.signal_axes[0].axis, dtype=float)
        counts = np.asarray(signal.data, dtype=float)
        title = getattr(signal.metadata.General, "title", f"Spectrum {index + 1}")
        color = colors[index % len(colors)] if colors else None
        ax.plot(energy, counts, label=title, color=color)

    reference_index = strongest_signal_index(signals)
    reference_signal = signals[reference_index]
    reference_peak_fits = getattr(reference_signal, "_peak_fits", [])
    reference_color = colors[reference_index % len(colors)] if colors else "tab:orange"
    annotate_peaks(
        ax,
        reference_peak_fits,
        show_markers=False,
        color=reference_color,
    )

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title("Processed Spectra with Peak Labels")
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_peak_identification(signal, peak_fits, calibration=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    energy = np.asarray(signal.axes_manager.signal_axes[0].axis, dtype=float)
    counts = np.asarray(signal.data, dtype=float)

    ax.plot(energy, counts, color="tab:orange", label="Processed Spectrum")
    annotate_peaks(ax, peak_fits, show_markers=True)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    title = f"Peak Identification: {signal.metadata.General.title}"
    if calibration is not None:
        title = f"{title} (Ga calibrated)"
        labels = calibration["reference_labels"]
        energies = calibration["reference_energies"]
        positions = calibration["reference_positions"]
        ax.text(
            0.02,
            0.98,
            (
                f"{labels[0]} -> {energies[0]:.5f} keV at channel {positions[0]:.2f}\n"
                f"{labels[1]} -> {energies[1]:.5f} keV at channel {positions[1]:.2f}\n"
                f"offset = {calibration['offset']:.5f} keV, bin width = {calibration['scale']:.6f} keV"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
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
    max_peaks=20,
    min_rel_height=0.001,
    min_rel_prominence=0.0003,
    min_distance_points=1,
):
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

    plot_spectrum_comparison(energy, counts, signal, subtract_bg, background, bg_method)

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
        plot_peak_identification(signal, peak_fits, calibration=calibration)
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

    background = np.clip(background, 0, counts)
    corrected = np.clip(counts - background, 0, None)

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
    max_peaks=20,
    min_rel_height=0.001,
    min_rel_prominence=0.0003,
    min_distance_points=1,
):
    """Plot one or multiple EDS spectra with HyperSpy."""
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
        )
        for file in files
    ]

    if overlay:
        plot_overlay_with_peak_labels(signals)
    else:
        for signal in signals:
            signal.plot()

    return signals


def main():
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
        default=20,
        help="Maximum number of peaks to annotate (default: 20)"
    )
    parser.add_argument(
        "--min-rel-height",
        type=float,
        default=0.001,
        help="Minimum peak height relative to the spectrum maximum (default: 0.001)"
    )
    parser.add_argument(
        "--min-rel-prominence",
        type=float,
        default=0.0003,
        help="Minimum relative slope threshold for HyperSpy peak detection (default: 0.0003)"
    )
    parser.add_argument(
        "--min-distance-points",
        type=int,
        default=1,
        help="Minimum spacing between detected peaks in data points (default: 1)"
    )
    args = parser.parse_args()

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
