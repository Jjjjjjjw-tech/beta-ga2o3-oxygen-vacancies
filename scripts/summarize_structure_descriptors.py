#!/usr/bin/env python3
"""
Summarize structural descriptors for oxygen vacancies.

Inputs
------
analysis/relaxation/tables/local_relaxation_details.csv
analysis/local_structure/first_shell_ga.csv
analysis/local_structure/local_distortion_by_ga.csv
analysis/local_structure/local_distortion_by_vacancy.csv
analysis/local_structure/ga_bond_change_summary.csv

Output
------
analysis/local_structure/structure_descriptors.csv

Each output row corresponds to one oxygen-vacancy configuration.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


# ============================================================
# File paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RELAXATION_ROOT = (
    PROJECT_ROOT
    / "analysis"
    / "relaxation"
)

LOCAL_STRUCTURE_ROOT = (
    PROJECT_ROOT
    / "analysis"
    / "local_structure"
)

RELAXATION_DETAILS_PATH = (
    RELAXATION_ROOT
    / "tables"
    / "local_relaxation_details.csv"
)

FIRST_SHELL_GA_PATH = (
    LOCAL_STRUCTURE_ROOT
    / "first_shell_ga.csv"
)

LOCAL_DISTORTION_BY_GA_PATH = (
    LOCAL_STRUCTURE_ROOT
    / "local_distortion_by_ga.csv"
)

LOCAL_DISTORTION_BY_VACANCY_PATH = (
    LOCAL_STRUCTURE_ROOT
    / "local_distortion_by_vacancy.csv"
)

GA_BOND_SUMMARY_PATH = (
    LOCAL_STRUCTURE_ROOT
    / "ga_bond_change_summary.csv"
)

OUTPUT_PATH = (
    LOCAL_STRUCTURE_ROOT
    / "structure_descriptors.csv"
)


# ============================================================
# General helper functions
# ============================================================

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
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(
            f"The CSV file contains no data rows:\n{file_path}"
        )

    return rows


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


def mean(values: list[float]) -> float:
    """Return arithmetic mean."""

    if not values:
        raise ValueError(
            "Cannot calculate the mean of an empty list."
        )

    return sum(values) / len(values)


def root_mean_square(
    values: list[float],
) -> float:
    """Return sqrt(mean(x^2))."""

    if not values:
        raise ValueError(
            "Cannot calculate RMS of an empty list."
        )

    return math.sqrt(
        sum(value ** 2 for value in values)
        / len(values)
    )


# ============================================================
# Structure name handling
# ============================================================

def get_structure_names(
    vacancy_rows: list[dict[str, str]],
) -> list[str]:
    """Return sorted vacancy structure names."""

    structure_names = sorted(
        {
            row["structure"]
            for row in vacancy_rows
        }
    )

    if not structure_names:
        raise ValueError(
            "No vacancy structure names were found."
        )

    return structure_names


def format_vacancy_label(
    structure_name: str,
) -> str:
    """Convert vacancy_O1 to O1."""

    return structure_name.replace(
        "vacancy_",
        "",
    )


# ============================================================
# Displacement descriptors
# ============================================================

def summarize_displacement(
    relaxation_rows: list[dict[str, str]],
    structure_name: str,
) -> dict[str, object]:
    """
    Summarize displacement of first-shell Ga atoms.

    Only Ga rows belonging to the selected structure are used.
    """

    ga_rows = [
        row
        for row in relaxation_rows
        if (
            row["structure"] == structure_name
            and row["element"] == "Ga"
        )
    ]

    if not ga_rows:
        raise ValueError(
            f"{structure_name}: no Ga relaxation rows found."
        )

    # Depending on the exact CSV version, the displacement
    # column may have one of these names.
    possible_displacement_columns = (
        "displacement_A",
        "displacement_magnitude_A",
        "total_displacement_A",
    )

    displacement_column = next(
        (
            column
            for column in possible_displacement_columns
            if column in ga_rows[0]
        ),
        None,
    )

    if displacement_column is None:
        available_columns = ", ".join(
            ga_rows[0].keys()
        )

        raise KeyError(
            "Cannot identify the displacement column in "
            "local_relaxation_details.csv.\n"
            f"Available columns:\n{available_columns}"
        )

    displacement_values = [
        float(row[displacement_column])
        for row in ga_rows
        if row[displacement_column] != ""
    ]

    if not displacement_values:
        raise ValueError(
            f"{structure_name}: no displacement values found."
        )

    radial_values: list[float] = []

    possible_radial_columns = (
        "radial_displacement_A",
        "radial_displacement",
    )

    radial_column = next(
        (
            column
            for column in possible_radial_columns
            if column in ga_rows[0]
        ),
        None,
    )

    if radial_column is not None:
        radial_values = [
            float(row[radial_column])
            for row in ga_rows
            if row[radial_column] != ""
        ]

    return {
        "mean_ga_displacement_A":
            mean(displacement_values),
        "rms_ga_displacement_A":
            root_mean_square(
                displacement_values
            ),
        "maximum_ga_displacement_A":
            max(displacement_values),
        "mean_radial_ga_displacement_A":
            (
                mean(radial_values)
                if radial_values
                else None
            ),
        "maximum_absolute_radial_ga_displacement_A":
            (
                max(abs(value) for value in radial_values)
                if radial_values
                else None
            ),
    }


# ============================================================
# First-shell descriptors
# ============================================================

def summarize_first_shell(
    first_shell_rows: list[dict[str, str]],
    structure_name: str,
) -> dict[str, object]:
    """Summarize first-shell Ga information."""

    rows = [
        row
        for row in first_shell_rows
        if row["structure"] == structure_name
    ]

    if not rows:
        raise ValueError(
            f"{structure_name}: no first-shell Ga rows found."
        )

    distances = [
        float(row["distance_to_vacancy_A"])
        for row in rows
    ]

    return {
        "number_of_first_shell_ga":
            len(rows),
        "mean_first_shell_ga_distance_A":
            mean(distances),
        "minimum_first_shell_ga_distance_A":
            min(distances),
        "maximum_first_shell_ga_distance_A":
            max(distances),
        "shell_boundary_gap_A":
            float(
                rows[0]["shell_boundary_gap_A"]
            ),
    }


# ============================================================
# Bond descriptors
# ============================================================

def summarize_bonds(
    ga_bond_rows: list[dict[str, str]],
    structure_name: str,
) -> dict[str, object]:
    """Summarize Ga–O bond and coordination changes."""

    rows = [
        row
        for row in ga_bond_rows
        if row["structure"] == structure_name
    ]

    if not rows:
        raise ValueError(
            f"{structure_name}: no Ga bond summary rows found."
        )

    mean_abs_changes = [
        abs(
            float(
                row[
                    "mean_preserved_bond_change_A"
                ]
            )
        )
        for row in rows
        if row[
            "mean_preserved_bond_change_A"
        ] != ""
    ]

    maximum_abs_changes = [
        abs(
            float(
                row[
                    "largest_absolute_bond_change_A"
                ]
            )
        )
        for row in rows
        if row[
            "largest_absolute_bond_change_A"
        ] != ""
    ]

    total_lost_bonds = sum(
        int(row["number_of_lost_bonds"])
        for row in rows
    )

    total_formed_bonds = sum(
        int(row["number_of_formed_bonds"])
        for row in rows
    )

    total_coordination_loss = sum(
        int(
            row["pristine_coordination_number"]
        )
        - int(
            row["vacancy_coordination_number"]
        )
        for row in rows
    )

    tetrahedral_to_threefold = sum(
        1
        for row in rows
        if (
            int(
                row[
                    "pristine_coordination_number"
                ]
            ) == 4
            and int(
                row[
                    "vacancy_coordination_number"
                ]
            ) == 3
        )
    )

    octahedral_to_fivefold = sum(
        1
        for row in rows
        if (
            int(
                row[
                    "pristine_coordination_number"
                ]
            ) == 6
            and int(
                row[
                    "vacancy_coordination_number"
                ]
            ) == 5
        )
    )

    return {
        "mean_absolute_ga_o_bond_change_A":
            mean(mean_abs_changes),
        "maximum_absolute_ga_o_bond_change_A":
            max(maximum_abs_changes),
        "total_lost_ga_o_bonds":
            total_lost_bonds,
        "total_formed_ga_o_bonds":
            total_formed_bonds,
        "total_coordination_loss":
            total_coordination_loss,
        "number_of_4_to_3_ga":
            tetrahedral_to_threefold,
        "number_of_6_to_5_ga":
            octahedral_to_fivefold,
    }


# ============================================================
# Local distortion descriptors
# ============================================================

def find_vacancy_distortion_row(
    vacancy_rows: list[dict[str, str]],
    structure_name: str,
) -> dict[str, str]:
    """Return the vacancy-level distortion row."""

    matching_rows = [
        row
        for row in vacancy_rows
        if row["structure"] == structure_name
    ]

    if len(matching_rows) != 1:
        raise ValueError(
            f"{structure_name}: expected exactly one "
            "vacancy-level distortion row, found "
            f"{len(matching_rows)}."
        )

    return matching_rows[0]


def summarize_distortion(
    vacancy_rows: list[dict[str, str]],
    ga_rows: list[dict[str, str]],
    structure_name: str,
) -> dict[str, object]:
    """Collect bond- and angle-distortion descriptors."""

    vacancy_row = find_vacancy_distortion_row(
        vacancy_rows=vacancy_rows,
        structure_name=structure_name,
    )

    matching_ga_rows = [
        row
        for row in ga_rows
        if row["structure"] == structure_name
    ]

    if not matching_ga_rows:
        raise ValueError(
            f"{structure_name}: no Ga distortion rows found."
        )

    maximum_bond_change = max(
        float(
            row[
                "maximum_absolute_bond_change_A"
            ]
        )
        for row in matching_ga_rows
        if row[
            "maximum_absolute_bond_change_A"
        ] != ""
    )

    maximum_angle_change = max(
        float(
            row[
                "maximum_absolute_angle_change_deg"
            ]
        )
        for row in matching_ga_rows
        if row[
            "maximum_absolute_angle_change_deg"
        ] != ""
    )

    return {
        "mean_absolute_bond_distortion_A":
            float(
                vacancy_row[
                    "mean_of_ga_mean_absolute_bond_change_A"
                ]
            ),
        "mean_rms_bond_distortion_A":
            float(
                vacancy_row[
                    "mean_of_ga_rms_bond_change_A"
                ]
            ),
        "maximum_local_bond_distortion_A":
            maximum_bond_change,
        "mean_absolute_angle_distortion_deg":
            float(
                vacancy_row[
                    "mean_of_ga_mean_absolute_angle_change_deg"
                ]
            ),
        "mean_rms_angle_distortion_deg":
            float(
                vacancy_row[
                    "mean_of_ga_rms_angle_change_deg"
                ]
            ),
        "maximum_local_angle_distortion_deg":
            maximum_angle_change,
    }


# ============================================================
# Main summary
# ============================================================

def build_structure_descriptors(
    relaxation_rows: list[dict[str, str]],
    first_shell_rows: list[dict[str, str]],
    ga_distortion_rows: list[dict[str, str]],
    vacancy_distortion_rows: list[dict[str, str]],
    ga_bond_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Build one descriptor row per vacancy."""

    structure_names = get_structure_names(
        vacancy_distortion_rows
    )

    descriptor_rows: list[
        dict[str, object]
    ] = []

    for structure_name in structure_names:
        displacement = summarize_displacement(
            relaxation_rows=relaxation_rows,
            structure_name=structure_name,
        )

        first_shell = summarize_first_shell(
            first_shell_rows=first_shell_rows,
            structure_name=structure_name,
        )

        bonds = summarize_bonds(
            ga_bond_rows=ga_bond_rows,
            structure_name=structure_name,
        )

        distortion = summarize_distortion(
            vacancy_rows=
                vacancy_distortion_rows,
            ga_rows=ga_distortion_rows,
            structure_name=structure_name,
        )

        descriptor_rows.append(
            {
                "structure":
                    structure_name,
                "vacancy_label":
                    format_vacancy_label(
                        structure_name
                    ),
                **first_shell,
                **displacement,
                **bonds,
                **distortion,
            }
        )

    return descriptor_rows


# ============================================================
# Main program
# ============================================================

def main() -> None:
    """Generate the structural-descriptor table."""

    relaxation_rows = read_csv_rows(
        RELAXATION_DETAILS_PATH
    )

    first_shell_rows = read_csv_rows(
        FIRST_SHELL_GA_PATH
    )

    ga_distortion_rows = read_csv_rows(
        LOCAL_DISTORTION_BY_GA_PATH
    )

    vacancy_distortion_rows = read_csv_rows(
        LOCAL_DISTORTION_BY_VACANCY_PATH
    )

    ga_bond_rows = read_csv_rows(
        GA_BOND_SUMMARY_PATH
    )

    descriptor_rows = (
        build_structure_descriptors(
            relaxation_rows=relaxation_rows,
            first_shell_rows=first_shell_rows,
            ga_distortion_rows=
                ga_distortion_rows,
            vacancy_distortion_rows=
                vacancy_distortion_rows,
            ga_bond_rows=ga_bond_rows,
        )
    )

    write_csv(
        output_path=OUTPUT_PATH,
        rows=descriptor_rows,
    )

    print("=" * 78)
    print("Structure descriptors")
    print("=" * 78)

    for row in descriptor_rows:
        print()
        print(row["structure"])

        print(
            "  First-shell Ga: "
            f"{row['number_of_first_shell_ga']}"
        )

        print(
            "  Mean Ga displacement: "
            f"{row['mean_ga_displacement_A']:.4f} Å"
        )

        print(
            "  Maximum Ga displacement: "
            f"{row['maximum_ga_displacement_A']:.4f} Å"
        )

        print(
            "  Mean |Δr|: "
            f"{row['mean_absolute_bond_distortion_A']:.4f} Å"
        )

        print(
            "  Maximum |Δr|: "
            f"{row['maximum_local_bond_distortion_A']:.4f} Å"
        )

        print(
            "  Mean |Δθ|: "
            f"{row['mean_absolute_angle_distortion_deg']:.4f}°"
        )

        print(
            "  Maximum |Δθ|: "
            f"{row['maximum_local_angle_distortion_deg']:.4f}°"
        )

        print(
            "  Lost Ga-O bonds: "
            f"{row['total_lost_ga_o_bonds']}"
        )

    print()
    print("=" * 78)
    print("Output file")
    print("=" * 78)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()