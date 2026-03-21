from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import hyperspy.api as hs


BIN_WIDTH_KEV = 0.02


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


def build_eds_signal(filepath, bin_width_kev=BIN_WIDTH_KEV, calibrate=False, known_peaks=None, expected_peaks=None, subtract_bg=True, bg_method="asls", bg_degree=2):
    energy, counts = read_eds_txt(filepath)
    signal = hs.signals.Signal1D(np.asarray(counts, dtype=float))
    signal.metadata.General.title = Path(filepath).stem
    signal.metadata.Signal.quantity = "Counts"

    axis = signal.axes_manager.signal_axes[0]
    axis.name = "Energy"
    axis.units = "keV"

    if len(energy) > 1:
        axis.scale = energy[1] - energy[0]
    else:
        axis.scale = bin_width_kev

    axis.offset = energy[0]

    # Calibrate the signal if requested
    # if calibrate and known_peaks and expected_peaks:
    #     signal = calibrate_signal(signal, known_peaks, expected_peaks)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Overlay the original spectrum
    ax.plot(energy, counts, label="With Bremsstrahlung", color="black") #, linestyle="--"

    processed_label = "Original Spectrum"
    background = None

    # Subtract background if requested
    if subtract_bg:
        signal, background = subtract_background(signal, method=bg_method, degree=bg_degree)
        processed_label = f"Background Subtracted ({bg_method})"
        ax.plot(
            energy,
            background,
            label=f"Estimated Background ({bg_method})",
            color="tab:red",
            linestyle=":",
            linewidth=1.5,
        )

    # Overlay the processed spectrum on the same axes
    ax.plot(
        signal.axes_manager.signal_axes[0].axis,
        signal.data,
        label=processed_label,
        color="tab:orange",
        linestyle="--",
    )
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title("Spectrum Comparison")
    ax.legend()
    fig.tight_layout()
    plt.show()

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


def plot_eds(files, overlay=True, bin_width_kev=BIN_WIDTH_KEV, subtract_bg=True, bg_method="asls", bg_degree=2):
    """Plot one or multiple EDS spectra with HyperSpy."""
    signals = [
        build_eds_signal(
            file,
            bin_width_kev=bin_width_kev,
            subtract_bg=subtract_bg,
            bg_method=bg_method,
            bg_degree=bg_degree
        )
        for file in files
    ]

    if overlay:
        hs.plot.plot_spectra(signals, style="overlap", legend="auto")
    else:
        for signal in signals:
            signal.plot()

    plt.show()

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

    args = parser.parse_args()

    plot_eds(
        args.files,
        overlay=not args.separate,
        bin_width_kev=args.bin_width,
        subtract_bg=args.subtract_bg,
        bg_method=args.bg_method,
        bg_degree=args.bg_degree
    )


if __name__ == "__main__":
    main()
