#!/usr/bin/env python3
"""
Analyze O-Ga-O bond-angle changes around oxygen vacancies.

The script:

1. Reads first-shell Ga atoms from first_shell_ga.csv.
2. Reads preserved Ga-O bonds from ga_o_bond_changes.csv.
3. Reuses atom mapping from local_relaxation_details.csv.
4. Calculates corresponding O-Ga-O angles in:
   - relaxed pristine CONTCAR
   - relaxed vacancy CONTCAR
5. Calculates angle changes:
       delta_angle = vacancy_angle - pristine_angle
6. Writes individual and combined CSV files.
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

from pymatgen.core import Structure


# ============================================================
# Configuration
# ============================================================

RESULT_DATE = "2026-07-25_results"

VACANCY_CASES = (
    "vacancy_O1",
    "vacancy_O2",
    "vacancy_O3",
)


# ============================================================
# File paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_ROOT = (
    PROJECT_ROOT
    / "beta-Ga2O3_oxygen_vacancy_VASP_inputs"
    / "received_results"
    / RESULT_DATE
)

LOCAL_STRUCTURE_ROOT = (
    PROJECT_ROOT
    / "analysis"
    / "local_structure"
)

RELAXATION_DETAILS_PATH = (
    PROJECT_ROOT
    / "analysis"
    / "relaxation"
    / "tables"
    / "local_relaxation_details.csv"
)

FIRST_SHELL_GA_PATH = (
    LOCAL_STRUCTURE_ROOT
    / "first_shell_ga.csv"
)

PRISTINE_CONTCAR_PATH = (
    RESULTS_ROOT
    / "pristine"
    / "relax"
    / "CONTCAR"
)

COMBINED_OUTPUT_PATH = (
    LOCAL_STRUCTURE_ROOT
    / "bond_angle_change_summary.csv"
)


# ============================================================
# General helper functions
# ============================================================

def load_structure(file_path: Path) -> Structure:
    """Load a POSCAR or CONTCAR structure."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Cannot find structure file:\n{file_path}"
        )

    return Structure.from_file(file_path)


def read_csv_rows(
    file_path: Path,
) -> list[dict[str, str]]:
    """Read a CSV file as a list of dictionaries."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Cannot find CSV file:\n{file_path}"
        )

    with file_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write dictionaries to a CSV file."""

    if not rows:
        raise ValueError(
            f"No rows available for {output_path.name}."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Atom mapping
# ============================================================

def build_reference_to_defect_mapping(
    relaxation_rows: list[dict[str, str]],
    case_name: str,
) -> dict[int, int]:
    """
    Build a zero-based reference-to-defect atom mapping.

    local_relaxation_details.csv stores one-based atom numbers:

        defect_atom_number
        reference_atom_number

    This function converts them to Python indices and reverses
    the mapping:

        reference index -> defect index
    """

    reference_to_defect: dict[int, int] = {}

    for row in relaxation_rows:
        if row["structure"] != case_name:
            continue

        defect_index = (
            int(row["defect_atom_number"]) - 1
        )

        reference_index = (
            int(row["reference_atom_number"]) - 1
        )

        reference_to_defect[reference_index] = defect_index

    if not reference_to_defect:
        raise ValueError(
            f"{case_name}: no atom mapping was found."
        )

    return reference_to_defect


# ============================================================
# First-shell Ga selection
# ============================================================

def get_first_shell_ga_rows(
    first_shell_rows: list[dict[str, str]],
    case_name: str,
) -> list[dict[str, str]]:
    """Return first-shell Ga records for one vacancy."""

    case_rows = [
        row
        for row in first_shell_rows
        if row["structure"] == case_name
    ]

    case_rows.sort(
        key=lambda row: int(
            row["first_shell_rank"]
        )
    )

    if not case_rows:
        raise ValueError(
            f"{case_name}: no first-shell Ga atoms found."
        )

    return case_rows


# ============================================================
# Preserved oxygen neighbors
# ============================================================

def get_preserved_oxygen_numbers(
    bond_rows: list[dict[str, str]],
    reference_ga_number: int,
    defect_ga_number: int,
) -> list[int]:
    """
    Return reference oxygen atom numbers for preserved Ga-O bonds.

    Only bonds marked as 'preserved' are used to construct
    comparable O-Ga-O angles.
    """

    oxygen_numbers = [
        int(row["reference_oxygen_atom_number"])
        for row in bond_rows
        if (
            int(row["reference_ga_atom_number"])
            == reference_ga_number
            and int(row["defect_ga_atom_number"])
            == defect_ga_number
            and row["bond_status"] == "preserved"
        )
    ]

    oxygen_numbers = sorted(set(oxygen_numbers))

    if len(oxygen_numbers) < 2:
        raise ValueError(
            f"Ga{reference_ga_number}: fewer than two "
            "preserved oxygen neighbors were found."
        )

    return oxygen_numbers


def generate_oxygen_pairs(
    oxygen_numbers: list[int],
) -> list[tuple[int, int]]:
    """Generate all unique oxygen pairs."""

    return list(combinations(oxygen_numbers, 2))


# ============================================================
# Angle calculation
# ============================================================

def calculate_angle_change(
    pristine: Structure,
    vacancy: Structure,
    reference_to_defect: dict[int, int],
    reference_ga_number: int,
    defect_ga_number: int,
    reference_o1_number: int,
    reference_o2_number: int,
) -> dict[str, object]:
    """
    Calculate one corresponding O-Ga-O angle in two structures.
    """

    reference_ga_index = reference_ga_number - 1
    defect_ga_index = defect_ga_number - 1

    reference_o1_index = reference_o1_number - 1
    reference_o2_index = reference_o2_number - 1

    if reference_o1_index not in reference_to_defect:
        raise KeyError(
            f"Reference O{reference_o1_number} has no "
            "corresponding defect atom."
        )

    if reference_o2_index not in reference_to_defect:
        raise KeyError(
            f"Reference O{reference_o2_number} has no "
            "corresponding defect atom."
        )

    defect_o1_index = reference_to_defect[
        reference_o1_index
    ]

    defect_o2_index = reference_to_defect[
        reference_o2_index
    ]

    # Optional consistency check for the Ga mapping.
    mapped_defect_ga_index = reference_to_defect.get(
        reference_ga_index
    )

    if (
        mapped_defect_ga_index is not None
        and mapped_defect_ga_index != defect_ga_index
    ):
        raise ValueError(
            f"Ga mapping mismatch: reference Ga"
            f"{reference_ga_number} maps to defect atom "
            f"{mapped_defect_ga_index + 1}, not "
            f"Ga{defect_ga_number}."
        )

    pristine_angle = pristine.get_angle(
        reference_o1_index,
        reference_ga_index,
        reference_o2_index,
    )

    vacancy_angle = vacancy.get_angle(
        defect_o1_index,
        defect_ga_index,
        defect_o2_index,
    )

    angle_change = vacancy_angle - pristine_angle

    return {
        "reference_oxygen_1_atom_number":
            reference_o1_number,
        "reference_ga_atom_number":
            reference_ga_number,
        "reference_oxygen_2_atom_number":
            reference_o2_number,
        "defect_oxygen_1_atom_number":
            defect_o1_index + 1,
        "defect_ga_atom_number":
            defect_ga_number,
        "defect_oxygen_2_atom_number":
            defect_o2_index + 1,
        "pristine_angle_deg":
            pristine_angle,
        "vacancy_angle_deg":
            vacancy_angle,
        "angle_change_deg":
            angle_change,
        "absolute_angle_change_deg":
            abs(angle_change),
    }


# ============================================================
# Analyze one vacancy
# ============================================================

def analyze_case(
    case_name: str,
    pristine: Structure,
    relaxation_rows: list[dict[str, str]],
    first_shell_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Analyze bond-angle changes for one vacancy."""

    vacancy_contcar_path = (
        RESULTS_ROOT
        / case_name
        / "relax"
        / "CONTCAR"
    )

    bond_changes_path = (
        LOCAL_STRUCTURE_ROOT
        / case_name
        / "ga_o_bond_changes.csv"
    )

    vacancy = load_structure(
        vacancy_contcar_path
    )

    bond_rows = read_csv_rows(
        bond_changes_path
    )

    case_ga_rows = get_first_shell_ga_rows(
        first_shell_rows=first_shell_rows,
        case_name=case_name,
    )

    reference_to_defect = (
        build_reference_to_defect_mapping(
            relaxation_rows=relaxation_rows,
            case_name=case_name,
        )
    )

    case_results: list[dict[str, object]] = []

    print("\n" + "=" * 78)
    print(case_name)
    print("=" * 78)

    for ga_row in case_ga_rows:
        reference_ga_number = int(
            ga_row["reference_ga_atom_number"]
        )

        defect_ga_number = int(
            ga_row["defect_ga_atom_number"]
        )

        preserved_oxygen_numbers = (
            get_preserved_oxygen_numbers(
                bond_rows=bond_rows,
                reference_ga_number=
                    reference_ga_number,
                defect_ga_number=
                    defect_ga_number,
            )
        )

        oxygen_pairs = generate_oxygen_pairs(
            preserved_oxygen_numbers
        )

        print()
        print(
            f"Reference Ga{reference_ga_number} "
            f"-> defect Ga{defect_ga_number}"
        )

        print(
            "Preserved O neighbors: "
            + ", ".join(
                f"O{number}"
                for number in preserved_oxygen_numbers
            )
        )

        print(
            f"Number of comparable angles: "
            f"{len(oxygen_pairs)}"
        )

        for reference_o1_number, reference_o2_number in (
            oxygen_pairs
        ):
            angle_result = calculate_angle_change(
                pristine=pristine,
                vacancy=vacancy,
                reference_to_defect=
                    reference_to_defect,
                reference_ga_number=
                    reference_ga_number,
                defect_ga_number=
                    defect_ga_number,
                reference_o1_number=
                    reference_o1_number,
                reference_o2_number=
                    reference_o2_number,
            )

            result_row = {
                "structure": case_name,
                "first_shell_rank": int(
                    ga_row["first_shell_rank"]
                ),
                "ga_distance_to_vacancy_A": float(
                    ga_row["distance_to_vacancy_A"]
                ),
                **angle_result,
            }

            case_results.append(result_row)

            print(
                f"  O{reference_o1_number}"
                f"-Ga{reference_ga_number}"
                f"-O{reference_o2_number}: "
                f"{angle_result['pristine_angle_deg']:.4f}° "
                f"-> "
                f"{angle_result['vacancy_angle_deg']:.4f}°, "
                f"Δ = "
                f"{angle_result['angle_change_deg']:+.4f}°"
            )

    return case_results


# ============================================================
# Main program
# ============================================================

def main() -> None:
    """Analyze bond-angle changes for all vacancy cases."""

    pristine = load_structure(
        PRISTINE_CONTCAR_PATH
    )

    relaxation_rows = read_csv_rows(
        RELAXATION_DETAILS_PATH
    )

    first_shell_rows = read_csv_rows(
        FIRST_SHELL_GA_PATH
    )

    all_results: list[dict[str, object]] = []

    for case_name in VACANCY_CASES:
        case_results = analyze_case(
            case_name=case_name,
            pristine=pristine,
            relaxation_rows=relaxation_rows,
            first_shell_rows=first_shell_rows,
        )

        case_output_path = (
            LOCAL_STRUCTURE_ROOT
            / case_name
            / "o_ga_o_angle_changes.csv"
        )

        write_csv(
            output_path=case_output_path,
            rows=case_results,
        )

        all_results.extend(case_results)

    write_csv(
        output_path=COMBINED_OUTPUT_PATH,
        rows=all_results,
    )

    print("\n" + "=" * 78)
    print("Output files")
    print("=" * 78)

    for case_name in VACANCY_CASES:
        print(
            LOCAL_STRUCTURE_ROOT
            / case_name
            / "o_ga_o_angle_changes.csv"
        )

    print(COMBINED_OUTPUT_PATH)

 
if __name__ == "__main__":
    main()

