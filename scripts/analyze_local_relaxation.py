#!/usr/bin/env python3
"""
Analyze local structural relaxation around oxygen vacancies.

The script:

1. Identifies the oxygen atom removed from a pristine reference structure.
2. Finds atoms nearest to the vacancy in the initial defect structure.
3. Calculates their atomic displacement during relaxation.
4. Determines whether each atom moves toward or away from the vacancy.
5. Exports detailed CSV tables.

The pristine reference must be the structure from which the vacancy POSCARs
were originally generated.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.util.coord import pbc_diff
from scipy.optimize import linear_sum_assignment


DEFAULT_CASES = ("vacancy_O1", "vacancy_O2", "vacancy_O3")


def load_structure(file_path: Path) -> Structure:
    """
    Load a structure from a POSCAR or CONTCAR file.

    Parameters
    ----------
    file_path
        Path to the structure file.

    Returns
    -------
    pymatgen.core.Structure
        Loaded structure.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Structure file not found: {file_path}")

    return Structure.from_file(file_path)


def check_lattice_consistency(
    reference: Structure,
    defect: Structure,
    case_name: str,
    tolerance: float = 1.0e-5,
) -> None:
    """
    Check that reference and defect initial structures use the same lattice.
    """
    maximum_difference = np.max(
        np.abs(reference.lattice.matrix - defect.lattice.matrix)
    )

    if maximum_difference > tolerance:
        raise ValueError(
            f"{case_name}: pristine and vacancy initial lattices differ by "
            f"up to {maximum_difference:.6f} Å."
        )


def match_species_sites(
    reference: Structure,
    defect: Structure,
    element: str,
) -> tuple[dict[int, int], list[int]]:
    """
    Match atoms of one element between reference and defect structures.

    Matching is performed globally using the Hungarian assignment algorithm
    and periodic-boundary distances.

    Parameters
    ----------
    reference
        Pristine reference structure.
    defect
        Initial vacancy structure.
    element
        Chemical symbol to match.

    Returns
    -------
    defect_to_reference
        Mapping from defect atom index to pristine atom index.
    unmatched_reference
        Reference indices not matched to any defect atom.
    """
    reference_indices = [
        index
        for index, site in enumerate(reference)
        if site.specie.symbol == element
    ]

    defect_indices = [
        index
        for index, site in enumerate(defect)
        if site.specie.symbol == element
    ]

    if len(reference_indices) < len(defect_indices):
        raise ValueError(
            f"Defect structure contains more {element} atoms than reference."
        )

    reference_coordinates = reference.frac_coords[reference_indices]
    defect_coordinates = defect.frac_coords[defect_indices]

    distance_matrix = reference.lattice.get_all_distances(
        reference_coordinates,
        defect_coordinates,
    )

    reference_rows, defect_columns = linear_sum_assignment(distance_matrix)

    defect_to_reference: dict[int, int] = {}

    for reference_row, defect_column in zip(
        reference_rows,
        defect_columns,
        strict=True,
    ):
        reference_index = reference_indices[reference_row]
        defect_index = defect_indices[defect_column]

        defect_to_reference[defect_index] = reference_index

    matched_reference_indices = set(defect_to_reference.values())

    unmatched_reference = [
        index
        for index in reference_indices
        if index not in matched_reference_indices
    ]

    return defect_to_reference, unmatched_reference


def identify_oxygen_vacancy(
    reference: Structure,
    defect: Structure,
    case_name: str,
    matching_tolerance: float = 0.20,
) -> tuple[int, np.ndarray, dict[int, int]]:
    """
    Identify the oxygen site removed from the pristine structure.

    Parameters
    ----------
    reference
        Pristine structure used to generate the vacancy.
    defect
        Initial vacancy structure.
    case_name
        Vacancy case name.
    matching_tolerance
        Maximum allowed matched-site distance in Å.

    Returns
    -------
    vacancy_reference_index
        Atom index of the removed oxygen in the pristine structure.
    vacancy_fractional_coordinates
        Fractional coordinates of the vacancy.
    defect_to_reference
        Mapping between all remaining defect atoms and reference atoms.
    """
    defect_to_reference: dict[int, int] = {}
    unmatched_by_species: dict[str, list[int]] = {}

    elements = sorted(
        {
            site.specie.symbol
            for site in reference
        }
        | {
            site.specie.symbol
            for site in defect
        }
    )

    for element in elements:
        mapping, unmatched = match_species_sites(
            reference=reference,
            defect=defect,
            element=element,
        )

        defect_to_reference.update(mapping)
        unmatched_by_species[element] = unmatched

        for defect_index, reference_index in mapping.items():
            matched_distance = reference.lattice.get_distance_and_image(
                reference.frac_coords[reference_index],
                defect.frac_coords[defect_index],
            )[0]

            if matched_distance > matching_tolerance:
                raise ValueError(
                    f"{case_name}: matched {element} atoms differ by "
                    f"{matched_distance:.4f} Å, exceeding the tolerance."
                )

    unmatched_oxygen = unmatched_by_species.get("O", [])

    unmatched_nonoxygen = {
        element: indices
        for element, indices in unmatched_by_species.items()
        if element != "O" and indices
    }

    if unmatched_nonoxygen:
        raise ValueError(
            f"{case_name}: unmatched non-oxygen atoms found: "
            f"{unmatched_nonoxygen}"
        )

    if len(unmatched_oxygen) != 1:
        raise ValueError(
            f"{case_name}: expected exactly one missing oxygen, but found "
            f"{len(unmatched_oxygen)}."
        )

    vacancy_reference_index = unmatched_oxygen[0]
    vacancy_fractional_coordinates = reference.frac_coords[
        vacancy_reference_index
    ]

    return (
        vacancy_reference_index,
        vacancy_fractional_coordinates,
        defect_to_reference,
    )


def displacement_vectors(
    initial: Structure,
    relaxed: Structure,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate PBC-aware displacement vectors and remove global translation.

    Returns
    -------
    corrected_vectors
        Drift-corrected Cartesian displacement vectors in Å.
    drift_vector
        Mean translation removed from all atoms.
    """
    if len(initial) != len(relaxed):
        raise ValueError(
            "Initial and relaxed structures contain different atom counts."
        )

    initial_species = [site.specie.symbol for site in initial]
    relaxed_species = [site.specie.symbol for site in relaxed]

    if initial_species != relaxed_species:
        raise ValueError(
            "Atomic ordering differs between POSCAR and CONTCAR."
        )

    fractional_difference = pbc_diff(
        relaxed.frac_coords,
        initial.frac_coords,
    )

    raw_vectors = np.array(
        [
            relaxed.lattice.get_cartesian_coords(vector)
            for vector in fractional_difference
        ]
    )

    drift_vector = np.mean(raw_vectors, axis=0)
    corrected_vectors = raw_vectors - drift_vector

    return corrected_vectors, drift_vector


def vector_from_vacancy(
    lattice,
    atom_fractional_coordinates: np.ndarray,
    vacancy_fractional_coordinates: np.ndarray,
) -> np.ndarray:
    """
    Return the shortest Cartesian vector from vacancy to atom under PBC.
    """
    fractional_vector = pbc_diff(
        atom_fractional_coordinates,
        vacancy_fractional_coordinates,
    )

    return lattice.get_cartesian_coords(fractional_vector)


def analyze_case(
    reference: Structure,
    results_root: Path,
    case_name: str,
    cutoff: float,
    nearest_count: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """
    Analyze local relaxation around one oxygen vacancy.
    """
    relaxation_directory = results_root / case_name / "relax"

    initial = load_structure(relaxation_directory / "POSCAR")
    relaxed = load_structure(relaxation_directory / "CONTCAR")

    check_lattice_consistency(reference, initial, case_name)

    (
        vacancy_reference_index,
        vacancy_fractional_coordinates,
        defect_to_reference,
    ) = identify_oxygen_vacancy(
        reference=reference,
        defect=initial,
        case_name=case_name,
    )

    corrected_vectors, drift_vector = displacement_vectors(
        initial=initial,
        relaxed=relaxed,
    )

    records: list[dict[str, object]] = []

    for defect_index, initial_site in enumerate(initial):
        initial_radial_vector = vector_from_vacancy(
            lattice=initial.lattice,
            atom_fractional_coordinates=initial_site.frac_coords,
            vacancy_fractional_coordinates=vacancy_fractional_coordinates,
        )

        relaxed_radial_vector = vector_from_vacancy(
            lattice=relaxed.lattice,
            atom_fractional_coordinates=relaxed[defect_index].frac_coords,
            vacancy_fractional_coordinates=vacancy_fractional_coordinates,
        )

        initial_distance = np.linalg.norm(initial_radial_vector)
        relaxed_distance = np.linalg.norm(relaxed_radial_vector)

        displacement_vector = corrected_vectors[defect_index]
        displacement_magnitude = np.linalg.norm(displacement_vector)

        if initial_distance > 1.0e-12:
            outward_unit_vector = (
                initial_radial_vector / initial_distance
            )
            radial_projection = float(
                np.dot(displacement_vector, outward_unit_vector)
            )
        else:
            radial_projection = float("nan")

        distance_change = relaxed_distance - initial_distance

        if radial_projection < -1.0e-4:
            direction = "toward"
        elif radial_projection > 1.0e-4:
            direction = "away"
        else:
            direction = "mostly_tangential"

        records.append(
            {
                "structure": case_name,
                "defect_atom_number": defect_index + 1,
                "reference_atom_number":
                    defect_to_reference[defect_index] + 1,
                "element": initial_site.specie.symbol,
                "initial_distance_to_vacancy_A": initial_distance,
                "relaxed_distance_to_vacancy_A": relaxed_distance,
                "distance_change_A": distance_change,
                "total_displacement_A": displacement_magnitude,
                "radial_projection_A": radial_projection,
                "motion_relative_to_vacancy": direction,
                "dx_A": displacement_vector[0],
                "dy_A": displacement_vector[1],
                "dz_A": displacement_vector[2],
            }
        )

    records.sort(
        key=lambda row: float(row["initial_distance_to_vacancy_A"])
    )

    local_records = [
        record
        for record in records
        if float(record["initial_distance_to_vacancy_A"]) <= cutoff
    ]

    nearest_ga = [
        record
        for record in records
        if record["element"] == "Ga"
    ][:nearest_count]

    vacancy_site = reference[vacancy_reference_index]

    summary = {
        "structure": case_name,
        "vacancy_reference_atom_number": vacancy_reference_index + 1,
        "vacancy_element": vacancy_site.specie.symbol,
        "vacancy_frac_x": vacancy_fractional_coordinates[0],
        "vacancy_frac_y": vacancy_fractional_coordinates[1],
        "vacancy_frac_z": vacancy_fractional_coordinates[2],
        "atoms_within_cutoff": len(local_records),
        "ga_atoms_reported": len(nearest_ga),
        "global_drift_A": float(np.linalg.norm(drift_vector)),
    }

    print("\n" + "=" * 78)
    print(case_name)
    print("=" * 78)
    print(
        f"Removed oxygen          : reference atom "
        f"{vacancy_reference_index + 1}"
    )
    print(
        "Vacancy fractional site : "
        f"({vacancy_fractional_coordinates[0]:.6f}, "
        f"{vacancy_fractional_coordinates[1]:.6f}, "
        f"{vacancy_fractional_coordinates[2]:.6f})"
    )
    print(
        f"Removed global drift    : "
        f"{np.linalg.norm(drift_vector):.6f} Å"
    )

    print(f"\nNearest {len(nearest_ga)} Ga atoms")
    print("-" * 78)
    print(
        f"{'Rank':>4} "
        f"{'Defect atom':>12} "
        f"{'Ref. atom':>10} "
        f"{'Initial r (Å)':>14} "
        f"{'Disp. (Å)':>11} "
        f"{'Radial (Å)':>12} "
        f"{'Direction':>12}"
    )

    for rank, record in enumerate(nearest_ga, start=1):
        print(
            f"{rank:>4} "
            f"{int(record['defect_atom_number']):>12} "
            f"{int(record['reference_atom_number']):>10} "
            f"{float(record['initial_distance_to_vacancy_A']):>14.6f} "
            f"{float(record['total_displacement_A']):>11.6f} "
            f"{float(record['radial_projection_A']):>12.6f} "
            f"{str(record['motion_relative_to_vacancy']):>12}"
        )

    print(f"\nAtoms initially within {cutoff:.2f} Å of vacancy")
    print("-" * 78)
    print(
        f"{'Atom':>6} "
        f"{'El.':>4} "
        f"{'Initial r':>12} "
        f"{'Final r':>12} "
        f"{'Δr':>10} "
        f"{'Disp.':>10}"
    )

    for record in local_records:
        print(
            f"{int(record['defect_atom_number']):>6} "
            f"{str(record['element']):>4} "
            f"{float(record['initial_distance_to_vacancy_A']):>12.6f} "
            f"{float(record['relaxed_distance_to_vacancy_A']):>12.6f} "
            f"{float(record['distance_change_A']):>10.6f} "
            f"{float(record['total_displacement_A']):>10.6f}"
        )

    return records, summary


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    """
    Write dictionaries to a CSV file.
    """
    if not rows:
        raise ValueError(f"No rows available for {output_path.name}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Identify oxygen vacancies and analyze local atomic relaxation."
        )
    )

    default_root = (
        Path("beta-Ga2O3_oxygen_vacancy_VASP_inputs")
        / "received_results"
        / "2026-07-25_results"
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_root,
        help="Directory containing pristine and vacancy calculation folders.",
    )

    parser.add_argument(
        "--pristine-reference",
        type=Path,
        default=None,
        help=(
            "Pristine POSCAR used to generate vacancy structures. "
            "Default: RESULTS_ROOT/pristine/relax/POSCAR"
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("analysis/relaxation/tables"),
        help="Directory for generated CSV files.",
    )

    parser.add_argument(
        "--cutoff",
        type=float,
        default=4.0,
        help="Local-neighborhood cutoff radius in Å.",
    )

    parser.add_argument(
        "--nearest-ga",
        type=int,
        default=6,
        help="Number of nearest Ga atoms displayed.",
    )

    return parser.parse_args()


def main() -> int:
    """
    Run local relaxation analysis.
    """
    arguments = parse_arguments()

    pristine_reference_path = arguments.pristine_reference

    if pristine_reference_path is None:
        pristine_reference_path = (
            arguments.results_root
            / "pristine"
            / "relax"
            / "POSCAR"
        )

    if arguments.cutoff <= 0:
        print("Error: --cutoff must be positive.", file=sys.stderr)
        return 1

    if arguments.nearest_ga < 1:
        print("Error: --nearest-ga must be at least 1.", file=sys.stderr)
        return 1

    try:
        reference = load_structure(pristine_reference_path)

        all_records: list[dict[str, object]] = []
        all_summaries: list[dict[str, object]] = []

        for case_name in DEFAULT_CASES:
            records, summary = analyze_case(
                reference=reference,
                results_root=arguments.results_root,
                case_name=case_name,
                cutoff=arguments.cutoff,
                nearest_count=arguments.nearest_ga,
            )

            all_records.extend(records)
            all_summaries.append(summary)

        detail_output = (
            arguments.output_directory
            / "local_relaxation_details.csv"
        )

        summary_output = (
            arguments.output_directory
            / "vacancy_site_summary.csv"
        )

        write_csv(detail_output, all_records)
        write_csv(summary_output, all_summaries)

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
    ) as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 1

    print("\n" + "=" * 78)
    print("Output files")
    print("=" * 78)
    print(detail_output)
    print(summary_output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())