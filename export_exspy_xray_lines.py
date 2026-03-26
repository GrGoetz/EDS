from pathlib import Path
import argparse
import csv

from exspy.material import elements
from exspy.utils.eds import get_xray_lines_near_energy


DEFAULT_OUTPUT = Path("exspy_xray_lines.csv")


def iter_xray_line_rows():
    all_elements = elements.as_dictionary()

    for element, element_data in sorted(all_elements.items()):
        if not isinstance(element_data, dict):
            continue

        atomic_properties = element_data.get("Atomic_properties", {})
        xray_lines = atomic_properties.get("Xray_lines", {})
        if not isinstance(xray_lines, dict):
            continue

        for line, line_data in sorted(xray_lines.items()):
            if not isinstance(line_data, dict):
                continue

            energy_kev = line_data.get("energy (keV)")
            weight = line_data.get("weight")
            if energy_kev is None:
                continue

            yield {
                "element": element,
                "line": line,
                "xray_line": f"{element}_{line}",
                "energy_kev": float(energy_kev),
                "weight": "" if weight is None else float(weight),
            }


def export_xray_lines(output_path):
    rows = list(iter_xray_line_rows())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["element", "line", "xray_line", "energy_kev", "weight"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def print_near_energy(energies, width):
    for energy in energies:
        matches = get_xray_lines_near_energy(energy, width=width)
        print(f"{energy:.4f} keV")
        if matches:
            for match in matches:
                print(f"  {match}")
        else:
            print("  no nearby lines found")


def main():
    parser = argparse.ArgumentParser(
        description="Export eXSpy X-ray transition energies and inspect nearby lines."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--near-energy",
        type=float,
        nargs="*",
        default=[],
        help="One or more energies in keV to check against the eXSpy line database.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=0.05,
        help="Half-width in keV for the nearby-line search (default: 0.05).",
    )
    args = parser.parse_args()

    row_count = export_xray_lines(args.output)
    print(f"Wrote {row_count} X-ray lines to {args.output}")

    if args.near_energy:
        print_near_energy(args.near_energy, width=args.width)


if __name__ == "__main__":
    main()
