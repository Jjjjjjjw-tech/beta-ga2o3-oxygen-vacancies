#!/usr/bin/env python3
"""
Analyze atomic displacements during VASP structural relaxation.

For each vacancy structure, this script compares the initial POSCAR with
the relaxed CONTCAR, calculates atomic displacements under periodic boundary
conditions, prints a summary, and exports CSV tables.

Outputs
-------
tables/atomic_displacements.csv
    Displacement information for every atom.

tables/relaxation_summary.csv
    Summary statistics for each vacancy structure.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.util.coord import pbc_diff


DEFAULT_CASES = ("vacancy_O1", "vacancy_O2", "vacancy_O3")


def load_structure(file_path: Path) -> Structure:
    """
    Load a crystal structure from a VASP structure file.

    Parameters
    ----------
    file_path
        Path to a POSCAR or CONTCAR file.

    Returns
    -------
    pymatgen.core.Structure
        Loaded crystal structure.

    Raises
    ------
    FileNotFoundError
        If the requested structure file does not exist.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Structure file not found: {file_path}")

    return Structure.from_file(file_path)


def validate_structure_pair(
    initial: Structure,
    relaxed: Structure,
    case_name: str,
) -> None:
    """
    Confirm that the initial and relaxed structures can be compared directly.

    The function checks atom count, atom ordering, and lattice consistency.

    Parameters
    ----------
    initial
        Initial structure from POSCAR.
    relaxed
        Relaxed structure from CONTCAR.
    case_name
        Name of the vacancy structure.
    """
    if len(initial) != len(relaxed):
        raise ValueError(
            f"{case_name}: POSCAR has {len(initial)} atoms, but "
            f"CONTCAR has {len(relaxed)} atoms."
        )

    initial_species = [str(site.specie) for site in initial]
    relaxed_species = [str(site.specie) for site in relaxed]

    if initial_species != relaxed_species:
        raise ValueError(
            f"{case_name}: atomic species or atom ordering changed between "
            "POSCAR and CONTCAR."
        )

    lattice_difference = np.max(
        np.abs(initial.lattice.matrix - relaxed.lattice.matrix)
    )

    if lattice_difference > 1.0e-5:
        print(
            f"Warning: {case_name} lattice changed by up to "
            f"{lattice_difference:.6f} Å."
        )


def calculate_displacements(
    initial: Structure,
    relaxed: Structure,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate atomic displacement vectors with periodic boundary conditions.

    Two displacement values are returned:

    1. Raw displacement:
       Direct movement from POSCAR to CONTCAR under periodic boundaries.

    2. Drift-corrected displacement:
       Raw displacement after removing the common rigid translation of the
       complete structure.

    Parameters
    ----------
    initial
        Initial structure.
    relaxed
        Relaxed structure.

    Returns
    -------
    raw_vectors
        Cartesian displacement vectors in Å.
    corrected_vectors
        Cartesian displacement vectors after removing global drift.
    drift_vector
        Average rigid translation vector removed from all atoms.
    """
    fractional_displacements = pbc_diff(
        relaxed.frac_coords,
        initial.frac_coords,
    )

    raw_vectors = np.array(
        [
            relaxed.lattice.get_cartesian_coords(frac_vector)
            for frac_vector in fractional_displacements
        ]
    )

    drift_vector = np.mean(raw_vectors, axis=0)
    corrected_vectors = raw_vectors - drift_vector

    return raw_vectors, corrected_vectors, drift_vector


def analyze_case(
    results_root: Path,
    case_name: str,
    top_n: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """
    Analyze one vacancy relaxation calculation.

    Parameters
    ----------
    results_root
        Directory containing vacancy_O1, vacancy_O2, and vacancy_O3.
    case_name
        Name of the vacancy calculation.
    top_n
        Number of most-displaced atoms to display.

    Returns
    -------
    atom_rows
        Per-atom displacement records.
    summary
        Summary statistics for the vacancy structure.
    """
    relaxation_directory = results_root / case_name / "relax"
    poscar_path = relaxation_directory / "POSCAR"
    contcar_path = relaxation_directory / "CONTCAR"

    initial = load_structure(poscar_path)
    relaxed = load_structure(contcar_path)

    validate_structure_pair(initial, relaxed, case_name)

    raw_vectors, corrected_vectors, drift_vector = calculate_displacements(
        initial,
        relaxed,
    )

    raw_magnitudes = np.linalg.norm(raw_vectors, axis=1)
    corrected_magnitudes = np.linalg.norm(corrected_vectors, axis=1)

    atom_rows: list[dict[str, object]] = []

    for atom_index, site in enumerate(initial):
        raw_vector = raw_vectors[atom_index]
        corrected_vector = corrected_vectors[atom_index]

        atom_rows.append(
            {
                "structure": case_name,
                "atom_number": atom_index + 1,
                "element": str(site.specie),
                "raw_displacement_A": raw_magnitudes[atom_index],
                "corrected_displacement_A": corrected_magnitudes[atom_index],
                "dx_A": corrected_vector[0],
                "dy_A": corrected_vector[1],
                "dz_A": corrected_vector[2],
                "initial_frac_x": initial.frac_coords[atom_index][0],
                "initial_frac_y": initial.frac_coords[atom_index][1],
                "initial_frac_z": initial.frac_coords[atom_index][2],
                "relaxed_frac_x": relaxed.frac_coords[atom_index][0],
                "relaxed_frac_y": relaxed.frac_coords[atom_index][1],
                "relaxed_frac_z": relaxed.frac_coords[atom_index][2],
            }
        )

    summary: dict[str, object] = {
        "structure": case_name,
        "number_of_atoms": len(initial),
        "maximum_displacement_A": float(np.max(corrected_magnitudes)),
        "mean_displacement_A": float(np.mean(corrected_magnitudes)),
        "median_displacement_A": float(np.median(corrected_magnitudes)),
        "rms_displacement_A": float(
            np.sqrt(np.mean(corrected_magnitudes**2))
        ),
        "drift_x_A": float(drift_vector[0]),
        "drift_y_A": float(drift_vector[1]),
        "drift_z_A": float(drift_vector[2]),
        "drift_magnitude_A": float(np.linalg.norm(drift_vector)),
    }

    sorted_indices = np.argsort(corrected_magnitudes)[::-1]

    print("\n" + "=" * 68)
    print(case_name)
    print("=" * 68)
    print(f"Number of atoms        : {len(initial)}")
    print(
        f"Removed global drift   : "
        f"{np.linalg.norm(drift_vector):.6f} Å"
    )
    print(
        f"Maximum displacement   : "
        f"{summary['maximum_displacement_A']:.6f} Å"
    )
    print(
        f"Mean displacement      : "
        f"{summary['mean_displacement_A']:.6f} Å"
    )
    print(
        f"Median displacement    : "
        f"{summary['median_displacement_A']:.6f} Å"
    )
    print(
        f"RMS displacement       : "
        f"{summary['rms_displacement_A']:.6f} Å"
    )

    print(f"\nTop {min(top_n, len(initial))} displaced atoms")
    print("-" * 68)
    print(
        f"{'Rank':>4}  {'Atom':>6}  {'Element':>7}  "
        f"{'Displacement (Å)':>18}"
    )

    for rank, atom_index in enumerate(sorted_indices[:top_n], start=1):
        print(
            f"{rank:>4}  "
            f"{atom_index + 1:>6}  "
            f"{str(initial[atom_index].specie):>7}  "
            f"{corrected_magnitudes[atom_index]:>18.6f}"
        )

    return atom_rows, summary


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    """
    Write a list of dictionaries to a CSV file.

    Parameters
    ----------
    output_path
        Destination CSV path.
    rows
        Records to export.
    """
    if not rows:
        raise ValueError(f"No data available for {output_path.name}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line options.
    """
    parser = argparse.ArgumentParser(
        description="Analyze VASP structural relaxation displacements."
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "beta-Ga2O3_oxygen_vacancy_VASP_inputs"
        )
        / "received_results"
        / "2026-07-25_results",
        help=(
            "Directory containing vacancy_O1, vacancy_O2, and vacancy_O3."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("analysis/relaxation/tables"),
        help="Directory in which CSV output files will be saved.",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of most-displaced atoms shown for each structure.",
    )

    return parser.parse_args()


def main() -> int:
    """
    Run structural relaxation analysis for all vacancy structures.

    Returns
    -------
    int
        Program exit status.
    """
    arguments = parse_arguments()

    if arguments.top < 1:
        print("Error: --top must be at least 1.", file=sys.stderr)
        return 1

    if not arguments.results_root.is_dir():
        print(
            f"Error: results directory not found:\n"
            f"  {arguments.results_root}",
            file=sys.stderr,
        )
        return 1

    all_atom_rows: list[dict[str, object]] = []
    all_summaries: list[dict[str, object]] = []

    try:
        for case_name in DEFAULT_CASES:
            atom_rows, summary = analyze_case(
                results_root=arguments.results_root,
                case_name=case_name,
                top_n=arguments.top,
            )

            all_atom_rows.extend(atom_rows)
            all_summaries.append(summary)

        displacement_output = (
            arguments.output_directory / "atomic_displacements.csv"
        )
        summary_output = (
            arguments.output_directory / "relaxation_summary.csv"
        )

        write_csv(displacement_output, all_atom_rows)
        write_csv(summary_output, all_summaries)

    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 1

    print("\n" + "=" * 68)
    print("Output files")
    print("=" * 68)
    print(displacement_output)
    print(summary_output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())