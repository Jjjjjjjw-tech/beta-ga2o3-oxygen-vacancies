#!/usr/bin/env python3
"""
Analyze local Ga–O bond changes around oxygen vacancies.

The script:

1. Reads atom-to-vacancy distances from local_relaxation_details.csv.
2. Detects the first-shell Ga atoms using a distance-gap criterion.
3. Compares relaxed pristine and relaxed vacancy structures.
4. Reports coordination-number and Ga–O bond-length changes.
5. Saves separate and combined CSV tables.
"""

from __future__ import annotations

import csv
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

# A gap of at least this size is treated as the boundary
# between the first and second Ga coordination shells.
FIRST_SHELL_GAP_THRESHOLD_A = 0.30

# At least this many Ga atoms must be considered before
# accepting a distance gap as the shell boundary.
MINIMUM_FIRST_SHELL_GA = 2

# Only inspect the first few nearest Ga atoms when searching
# for the first-shell boundary.
MAXIMUM_GA_CANDIDATES = 8

# Ga–O distances at or below this value are counted as bonds.
GA_O_BOND_CUTOFF_A = 2.50


# ============================================================
# File paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    PROJECT_ROOT
    / "beta-Ga2O3_oxygen_vacancy_VASP_inputs"
    / "received_results"
    / RESULT_DATE
)

PRISTINE_CONTCAR_PATH = (
    RESULTS_DIR
    / "pristine"
    / "relax"
    / "CONTCAR"
)

RELAXATION_DETAILS_PATH = (
    PROJECT_ROOT
    / "analysis"
    / "relaxation"
    / "tables"
    / "local_relaxation_details.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "analysis"
    / "local_structure"
)


# ============================================================
# General helper functions
# ============================================================

def load_structure(file_path: Path) -> Structure:
    """Load a POSCAR or CONTCAR file."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Cannot find structure file:\n{file_path}"
        )

    return Structure.from_file(file_path)


def read_csv_rows(
    file_path: Path,
) -> list[dict[str, str]]:
    """Read a CSV file into a list of dictionaries."""

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
# Detect first-shell Ga atoms
# ============================================================

def get_sorted_ga_rows(
    relaxation_rows: list[dict[str, str]],
    case_name: str,
) -> list[dict[str, str]]:
    """Return all Ga records sorted by distance to the vacancy."""

    ga_rows = [
        row
        for row in relaxation_rows
        if (
            row["structure"] == case_name
            and row["element"] == "Ga"
        )
    ]

    ga_rows.sort(
        key=lambda row: float(
            row["initial_distance_to_vacancy_A"]
        )
    )

    if not ga_rows:
        raise ValueError(
            f"{case_name}: no Ga atoms were found."
        )

    return ga_rows


def detect_first_shell_ga(
    relaxation_rows: list[dict[str, str]],
    case_name: str,
    gap_threshold: float,
    minimum_shell_size: int,
    maximum_candidates: int,
) -> tuple[
    list[dict[str, str]],
    float,
    float,
    float,
]:
    """
    Detect first-shell Ga atoms using the first significant distance gap.

    Returns
    -------
    first_shell_rows
        Ga records belonging to the detected first shell.
    shell_boundary_gap
        Distance gap separating the first and second shells.
    last_first_shell_distance
        Distance of the outermost first-shell Ga.
    first_second_shell_distance
        Distance of the nearest Ga after the first shell.
    """

    ga_rows = get_sorted_ga_rows(
        relaxation_rows=relaxation_rows,
        case_name=case_name,
    )

    candidate_count = min(
        maximum_candidates,
        len(ga_rows),
    )

    candidate_rows = ga_rows[:candidate_count]

    distances = [
        float(row["initial_distance_to_vacancy_A"])
        for row in candidate_rows
    ]

    if len(distances) <= minimum_shell_size:
        raise ValueError(
            f"{case_name}: too few Ga atoms to identify "
            "a shell boundary."
        )

    shell_end_index: int | None = None
    shell_boundary_gap: float | None = None

    # Example:
    #
    # distances = [1.87, 2.01, 2.08, 2.08, 2.62, ...]
    #
    # A gap is calculated between each pair:
    #
    # 2.62 - 2.08 = 0.54 Å
    #
    # The first gap above the threshold defines the shell boundary.
    for index in range(
        minimum_shell_size - 1,
        len(distances) - 1,
    ):
        current_distance = distances[index]
        next_distance = distances[index + 1]

        gap = next_distance - current_distance

        if gap >= gap_threshold:
            shell_end_index = index + 1
            shell_boundary_gap = gap
            break

    if (
        shell_end_index is None
        or shell_boundary_gap is None
    ):
        distance_text = ", ".join(
            f"{distance:.4f}"
            for distance in distances
        )

        raise ValueError(
            f"{case_name}: no Ga-shell distance gap of at least "
            f"{gap_threshold:.2f} Å was found among the first "
            f"{candidate_count} Ga atoms.\n"
            f"Distances: {distance_text}"
        )

    first_shell_rows = candidate_rows[:shell_end_index]

    last_first_shell_distance = distances[
        shell_end_index - 1
    ]

    first_second_shell_distance = distances[
        shell_end_index
    ]

    return (
        first_shell_rows,
        shell_boundary_gap,
        last_first_shell_distance,
        first_second_shell_distance,
    )


# ============================================================
# Mapping stored in relaxation CSV
# ============================================================

def build_defect_to_reference_mapping(
    relaxation_rows: list[dict[str, str]],
    case_name: str,
) -> dict[int, int]:
    """
    Build a zero-based defect-to-pristine atom-index mapping.

    The CSV stores one-based atom numbers, so one is subtracted
    to recover Python indices.
    """

    mapping: dict[int, int] = {}

    for row in relaxation_rows:
        if row["structure"] != case_name:
            continue

        defect_index = (
            int(row["defect_atom_number"]) - 1
        )

        reference_index = (
            int(row["reference_atom_number"]) - 1
        )

        mapping[defect_index] = reference_index

    if not mapping:
        raise ValueError(
            f"{case_name}: no atom mapping was found."
        )

    return mapping


# ============================================================
# Ga–O coordination analysis
# ============================================================

def get_oxygen_bonds(
    structure: Structure,
    ga_index: int,
    cutoff: float,
) -> dict[int, float]:
    """
    Return all O neighbors of one Ga within the cutoff.

    Keys
    ----
    Zero-based oxygen indices.

    Values
    ------
    Ga–O distances in Å.
    """

    ga_site = structure[ga_index]

    if ga_site.species_string != "Ga":
        raise ValueError(
            f"Atom {ga_index + 1} is "
            f"{ga_site.species_string}, not Ga."
        )

    oxygen_bonds: dict[int, float] = {}

    for oxygen_index, oxygen_site in enumerate(
        structure
    ):
        if oxygen_site.species_string != "O":
            continue

        distance = structure.get_distance(
            ga_index,
            oxygen_index,
        )

        if distance <= cutoff:
            oxygen_bonds[oxygen_index] = distance

    return oxygen_bonds


def compare_one_ga(
    case_name: str,
    pristine: Structure,
    vacancy: Structure,
    reference_ga_number: int,
    defect_ga_number: int,
    ga_distance_to_vacancy: float,
    defect_to_reference: dict[int, int],
) -> list[dict[str, object]]:
    """Compare the local oxygen environment of one Ga atom."""

    reference_ga_index = reference_ga_number - 1
    defect_ga_index = defect_ga_number - 1

    pristine_bonds = get_oxygen_bonds(
        structure=pristine,
        ga_index=reference_ga_index,
        cutoff=GA_O_BOND_CUTOFF_A,
    )

    vacancy_bonds_defect_indices = get_oxygen_bonds(
        structure=vacancy,
        ga_index=defect_ga_index,
        cutoff=GA_O_BOND_CUTOFF_A,
    )

    # Convert vacancy oxygen indices into pristine/reference
    # oxygen indices so that the same chemical bonds can be
    # compared between the two structures.
    vacancy_bonds_reference_indices: dict[
        int,
        float,
    ] = {}

    for (
        defect_oxygen_index,
        vacancy_distance,
    ) in vacancy_bonds_defect_indices.items():

        if defect_oxygen_index not in defect_to_reference:
            raise KeyError(
                f"{case_name}: defect atom "
                f"{defect_oxygen_index + 1} has no "
                "reference mapping."
            )

        reference_oxygen_index = (
            defect_to_reference[defect_oxygen_index]
        )

        vacancy_bonds_reference_indices[
            reference_oxygen_index
        ] = vacancy_distance

    all_oxygen_indices = sorted(
        set(pristine_bonds)
        | set(vacancy_bonds_reference_indices)
    )

    comparison_rows: list[dict[str, object]] = []

    for reference_oxygen_index in all_oxygen_indices:
        pristine_distance = pristine_bonds.get(
            reference_oxygen_index
        )

        vacancy_distance = (
            vacancy_bonds_reference_indices.get(
                reference_oxygen_index
            )
        )

        if (
            pristine_distance is not None
            and vacancy_distance is not None
        ):
            bond_status = "preserved"

            bond_length_change = (
                vacancy_distance - pristine_distance
            )

        elif pristine_distance is not None:
            bond_status = "lost"
            bond_length_change = None

        else:
            bond_status = "formed"
            bond_length_change = None

        comparison_rows.append(
            {
                "structure": case_name,
                "reference_ga_atom_number":
                    reference_ga_number,
                "defect_ga_atom_number":
                    defect_ga_number,
                "ga_initial_distance_to_vacancy_A":
                    ga_distance_to_vacancy,
                "reference_oxygen_atom_number":
                    reference_oxygen_index + 1,
                "bond_status": bond_status,
                "pristine_bond_length_A":
                    pristine_distance,
                "vacancy_bond_length_A":
                    vacancy_distance,
                "bond_length_change_A":
                    bond_length_change,
                "pristine_coordination_number":
                    len(pristine_bonds),
                "vacancy_coordination_number":
                    len(
                        vacancy_bonds_reference_indices
                    ),
                "ga_o_bond_cutoff_A":
                    GA_O_BOND_CUTOFF_A,
            }
        )

    return comparison_rows

def summarize_ga_bond_changes(
    bond_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """
    Summarize Ga–O bond changes for each first-shell Ga atom.

    Only preserved bonds are included when calculating
    mean, maximum, and minimum bond-length changes.
    """

    grouped_rows: dict[
        tuple[str, int, int],
        list[dict[str, object]],
    ] = {}

    for row in bond_rows:
        group_key = (
            str(row["structure"]),
            int(row["reference_ga_atom_number"]),
            int(row["defect_ga_atom_number"]),
        )

        grouped_rows.setdefault(
            group_key,
            [],
        ).append(row)

    summary_rows: list[dict[str, object]] = []

    for (
        structure_name,
        reference_ga_number,
        defect_ga_number,
    ), ga_rows in grouped_rows.items():

        preserved_changes = [
            float(row["bond_length_change_A"])
            for row in ga_rows
            if (
                row["bond_status"] == "preserved"
                and row["bond_length_change_A"] is not None
            )
        ]

        lost_oxygen_numbers = [
            int(row["reference_oxygen_atom_number"])
            for row in ga_rows
            if row["bond_status"] == "lost"
        ]

        formed_oxygen_numbers = [
            int(row["reference_oxygen_atom_number"])
            for row in ga_rows
            if row["bond_status"] == "formed"
        ]

        pristine_coordination = int(
            ga_rows[0]["pristine_coordination_number"]
        )

        vacancy_coordination = int(
            ga_rows[0]["vacancy_coordination_number"]
        )

        if preserved_changes:
            mean_change = (
                sum(preserved_changes)
                / len(preserved_changes)
            )

            maximum_change = max(preserved_changes)
            minimum_change = min(preserved_changes)
            maximum_absolute_change = max(
                preserved_changes,
                key=abs,
            )

        else:
            mean_change = None
            maximum_change = None
            minimum_change = None
            maximum_absolute_change = None

        summary_rows.append(
            {
                "structure": structure_name,
                "reference_ga_atom_number":
                    reference_ga_number,
                "defect_ga_atom_number":
                    defect_ga_number,
                "coordination_change":
                    f"{pristine_coordination} -> "
                    f"{vacancy_coordination}",
                "pristine_coordination_number":
                    pristine_coordination,
                "vacancy_coordination_number":
                    vacancy_coordination,
                "number_of_preserved_bonds":
                    len(preserved_changes),
                "number_of_lost_bonds":
                    len(lost_oxygen_numbers),
                "lost_oxygen_atom_numbers":
                    ";".join(
                        f"O{number}"
                        for number in lost_oxygen_numbers
                    ),
                "number_of_formed_bonds":
                    len(formed_oxygen_numbers),
                "formed_oxygen_atom_numbers":
                    ";".join(
                        f"O{number}"
                        for number in formed_oxygen_numbers
                    ),
                "mean_preserved_bond_change_A":
                    mean_change,
                "maximum_bond_elongation_A":
                    maximum_change,
                "maximum_bond_contraction_A":
                    minimum_change,
                "largest_absolute_bond_change_A":
                    maximum_absolute_change,
                "ga_initial_distance_to_vacancy_A":
                    float(
                        ga_rows[0][
                            "ga_initial_distance_to_vacancy_A"
                        ]
                    ),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            str(row["structure"]),
            float(
                row[
                    "ga_initial_distance_to_vacancy_A"
                ]
            ),
        )
    )

    return summary_rows


# ============================================================
# Analyze one vacancy case
# ============================================================

def analyze_case(
    case_name: str,
    pristine: Structure,
    relaxation_rows: list[dict[str, str]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Analyze all first-shell Ga atoms around one vacancy."""

    vacancy_path = (
        RESULTS_DIR
        / case_name
        / "relax"
        / "CONTCAR"
    )

    vacancy = load_structure(vacancy_path)

    (
        first_shell_ga_rows,
        shell_gap,
        outer_first_shell_distance,
        inner_second_shell_distance,
    ) = detect_first_shell_ga(
        relaxation_rows=relaxation_rows,
        case_name=case_name,
        gap_threshold=FIRST_SHELL_GAP_THRESHOLD_A,
        minimum_shell_size=MINIMUM_FIRST_SHELL_GA,
        maximum_candidates=MAXIMUM_GA_CANDIDATES,
    )

    defect_to_reference = (
        build_defect_to_reference_mapping(
            relaxation_rows=relaxation_rows,
            case_name=case_name,
        )
    )

    bond_rows: list[dict[str, object]] = []
    shell_rows: list[dict[str, object]] = []

    print("\n" + "=" * 78)
    print(case_name)
    print("=" * 78)

    print(
        f"Detected first-shell Ga atoms: "
        f"{len(first_shell_ga_rows)}"
    )

    print(
        f"Shell boundary: "
        f"{outer_first_shell_distance:.4f} Å "
        f"-> {inner_second_shell_distance:.4f} Å"
    )

    print(
        f"Boundary gap: {shell_gap:.4f} Å"
    )

    print("\nFirst-shell Ga atoms")
    print("-" * 78)

    for rank, ga_row in enumerate(
        first_shell_ga_rows,
        start=1,
    ):
        defect_ga_number = int(
            ga_row["defect_atom_number"]
        )

        reference_ga_number = int(
            ga_row["reference_atom_number"]
        )

        ga_distance = float(
            ga_row[
                "initial_distance_to_vacancy_A"
            ]
        )

        print(
            f"{rank:>2}. "
            f"defect Ga{defect_ga_number} "
            f"-> reference Ga{reference_ga_number}, "
            f"distance = {ga_distance:.4f} Å"
        )

        shell_rows.append(
            {
                "structure": case_name,
                "first_shell_rank": rank,
                "defect_ga_atom_number":
                    defect_ga_number,
                "reference_ga_atom_number":
                    reference_ga_number,
                "distance_to_vacancy_A":
                    ga_distance,
                "number_of_first_shell_ga":
                    len(first_shell_ga_rows),
                "outermost_first_shell_distance_A":
                    outer_first_shell_distance,
                "nearest_second_shell_distance_A":
                    inner_second_shell_distance,
                "shell_boundary_gap_A":
                    shell_gap,
                "gap_threshold_A":
                    FIRST_SHELL_GAP_THRESHOLD_A,
            }
        )

        ga_bond_rows = compare_one_ga(
            case_name=case_name,
            pristine=pristine,
            vacancy=vacancy,
            reference_ga_number=reference_ga_number,
            defect_ga_number=defect_ga_number,
            ga_distance_to_vacancy=ga_distance,
            defect_to_reference=defect_to_reference,
        )

        bond_rows.extend(ga_bond_rows)

        pristine_coordination = ga_bond_rows[0][
            "pristine_coordination_number"
        ]

        vacancy_coordination = ga_bond_rows[0][
            "vacancy_coordination_number"
        ]

        print(
            f"\nGa{reference_ga_number}: "
            f"coordination "
            f"{pristine_coordination} "
            f"-> {vacancy_coordination}"
        )

        for bond_row in ga_bond_rows:
            pristine_length = bond_row[
                "pristine_bond_length_A"
            ]

            vacancy_length = bond_row[
                "vacancy_bond_length_A"
            ]

            length_change = bond_row[
                "bond_length_change_A"
            ]

            pristine_text = (
                f"{pristine_length:.4f}"
                if pristine_length is not None
                else "—"
            )

            vacancy_text = (
                f"{vacancy_length:.4f}"
                if vacancy_length is not None
                else "—"
            )

            change_text = (
                f"{length_change:+.4f}"
                if length_change is not None
                else "—"
            )

            print(
                f"  O"
                f"{bond_row['reference_oxygen_atom_number']:>2}: "
                f"{pristine_text:>7} -> "
                f"{vacancy_text:>7} Å, "
                f"Δ = {change_text:>7} Å, "
                f"{bond_row['bond_status']}"
            )

    return bond_rows, shell_rows


# ============================================================
# Main program
# ============================================================

def main() -> None:
    """Analyze all configured oxygen vacancies."""

    pristine = load_structure(
        PRISTINE_CONTCAR_PATH
    )

    relaxation_rows = read_csv_rows(
        RELAXATION_DETAILS_PATH
    )

    all_bond_rows: list[dict[str, object]] = []
    all_shell_rows: list[dict[str, object]] = []

    for case_name in VACANCY_CASES:
        bond_rows, shell_rows = analyze_case(
            case_name=case_name,
            pristine=pristine,
            relaxation_rows=relaxation_rows,
        )

        case_output_path = (
            OUTPUT_ROOT
            / case_name
            / "ga_o_bond_changes.csv"
        )

        write_csv(
            output_path=case_output_path,
            rows=bond_rows,
        )

        all_bond_rows.extend(bond_rows)
        all_shell_rows.extend(shell_rows)

    combined_bond_output = (
        OUTPUT_ROOT
        / "bond_change_summary.csv"
    )

    first_shell_output = (
        OUTPUT_ROOT
        / "first_shell_ga.csv"
    )

    write_csv(
        output_path=combined_bond_output,
        rows=all_bond_rows,
    )

    write_csv(
        output_path=first_shell_output,
        rows=all_shell_rows,
    )
    ga_summary_rows = summarize_ga_bond_changes(
    all_bond_rows
    )

    ga_summary_output = (
    OUTPUT_ROOT
    / "ga_bond_change_summary.csv"
    )

    write_csv(
    output_path=ga_summary_output,
    rows=ga_summary_rows,
    )
    
    print("\n" + "=" * 78)
    print("Output files")
    print("=" * 78)

    for case_name in VACANCY_CASES:
        print(
            OUTPUT_ROOT
            / case_name
            / "ga_o_bond_changes.csv"
        )

    print(combined_bond_output)
    print(first_shell_output)


if __name__ == "__main__":
    main()