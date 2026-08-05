#!/usr/bin/env python3
"""
Summarize local structural distortion around oxygen vacancies.

Inputs
------
analysis/local_structure/bond_change_summary.csv
analysis/local_structure/bond_angle_change_summary.csv

Outputs
-------
analysis/local_structure/local_distortion_by_ga.csv
analysis/local_structure/local_distortion_by_vacancy.csv
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

LOCAL_STRUCTURE_ROOT = (
    PROJECT_ROOT
    / "analysis"
    / "local_structure"
)

BOND_CHANGE_PATH = (
    LOCAL_STRUCTURE_ROOT
    / "bond_change_summary.csv"
)

ANGLE_CHANGE_PATH = (
    LOCAL_STRUCTURE_ROOT
    / "bond_angle_change_summary.csv"
)

GA_OUTPUT_PATH = (
    LOCAL_STRUCTURE_ROOT
    / "local_distortion_by_ga.csv"
)

VACANCY_OUTPUT_PATH = (
    LOCAL_STRUCTURE_ROOT
    / "local_distortion_by_vacancy.csv"
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


def mean(values: list[float]) -> float:
    """Return the arithmetic mean."""

    if not values:
        raise ValueError("Cannot calculate the mean of an empty list.")

    return sum(values) / len(values)


def root_mean_square(values: list[float]) -> float:
    """Return sqrt(mean(x^2))."""

    if not values:
        raise ValueError(
            "Cannot calculate RMS for an empty list."
        )

    return math.sqrt(
        sum(value ** 2 for value in values)
        / len(values)
    )


# ============================================================
# Group input rows by vacancy and Ga
# ============================================================

def group_bond_rows(
    bond_rows: list[dict[str, str]],
) -> dict[
    tuple[str, int, int],
    list[dict[str, str]],
]:
    """
    Group preserved Ga-O bonds by vacancy and Ga.

    Key
    ---
    (
        structure,
        reference_ga_atom_number,
        defect_ga_atom_number,
    )
    """

    grouped: dict[
        tuple[str, int, int],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in bond_rows:
        if row["bond_status"] != "preserved":
            continue

        if not row["bond_length_change_A"]:
            continue

        key = (
            row["structure"],
            int(row["reference_ga_atom_number"]),
            int(row["defect_ga_atom_number"]),
        )

        grouped[key].append(row)

    return dict(grouped)


def group_angle_rows(
    angle_rows: list[dict[str, str]],
) -> dict[
    tuple[str, int, int],
    list[dict[str, str]],
]:
    """Group O-Ga-O angles by vacancy and Ga."""

    grouped: dict[
        tuple[str, int, int],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in angle_rows:
        key = (
            row["structure"],
            int(row["reference_ga_atom_number"]),
            int(row["defect_ga_atom_number"]),
        )

        grouped[key].append(row)

    return dict(grouped)


# ============================================================
# Per-Ga distortion statistics
# ============================================================

def summarize_by_ga(
    bond_rows: list[dict[str, str]],
    angle_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Calculate bond and angle distortion for each Ga atom."""

    grouped_bonds = group_bond_rows(bond_rows)
    grouped_angles = group_angle_rows(angle_rows)

    all_keys = sorted(
        set(grouped_bonds)
        | set(grouped_angles),
        key=lambda key: (
            key[0],
            key[1],
        ),
    )

    summary_rows: list[dict[str, object]] = []

    for key in all_keys:
        structure_name, reference_ga_number, defect_ga_number = key

        ga_bond_rows = grouped_bonds.get(key, [])
        ga_angle_rows = grouped_angles.get(key, [])

        bond_changes = [
            float(row["bond_length_change_A"])
            for row in ga_bond_rows
        ]

        angle_changes = [
            float(row["angle_change_deg"])
            for row in ga_angle_rows
        ]

        absolute_bond_changes = [
            abs(value)
            for value in bond_changes
        ]

        absolute_angle_changes = [
            abs(value)
            for value in angle_changes
        ]

        if ga_bond_rows:
            pristine_coordination = int(
                ga_bond_rows[0][
                    "pristine_coordination_number"
                ]
            )

            vacancy_coordination = int(
                ga_bond_rows[0][
                    "vacancy_coordination_number"
                ]
            )

            ga_distance_to_vacancy = float(
                ga_bond_rows[0][
                    "ga_initial_distance_to_vacancy_A"
                ]
            )

        else:
            pristine_coordination = None
            vacancy_coordination = None
            ga_distance_to_vacancy = None

        summary_rows.append(
            {
                "structure": structure_name,
                "reference_ga_atom_number":
                    reference_ga_number,
                "defect_ga_atom_number":
                    defect_ga_number,
                "coordination_change":
                    (
                        f"{pristine_coordination} -> "
                        f"{vacancy_coordination}"
                        if pristine_coordination is not None
                        else ""
                    ),
                "ga_initial_distance_to_vacancy_A":
                    ga_distance_to_vacancy,
                "number_of_preserved_bonds":
                    len(bond_changes),
                "mean_bond_change_A":
                    mean(bond_changes)
                    if bond_changes
                    else None,
                "mean_absolute_bond_change_A":
                    mean(absolute_bond_changes)
                    if absolute_bond_changes
                    else None,
                "rms_bond_change_A":
                    root_mean_square(bond_changes)
                    if bond_changes
                    else None,
                "maximum_absolute_bond_change_A":
                    max(absolute_bond_changes)
                    if absolute_bond_changes
                    else None,
                "number_of_comparable_angles":
                    len(angle_changes),
                "mean_angle_change_deg":
                    mean(angle_changes)
                    if angle_changes
                    else None,
                "mean_absolute_angle_change_deg":
                    mean(absolute_angle_changes)
                    if absolute_angle_changes
                    else None,
                "rms_angle_change_deg":
                    root_mean_square(angle_changes)
                    if angle_changes
                    else None,
                "maximum_absolute_angle_change_deg":
                    max(absolute_angle_changes)
                    if absolute_angle_changes
                    else None,
            }
        )

    return summary_rows


# ============================================================
# Vacancy-level statistics
# ============================================================

def summarize_by_vacancy(
    ga_summary_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """
    Average per-Ga distortion metrics for each vacancy.

    Each first-shell Ga contributes one value to the vacancy-level
    average, so a Ga with more preserved bonds or more angles does
    not automatically receive a larger weight.
    """

    grouped: dict[
        str,
        list[dict[str, object]],
    ] = defaultdict(list)

    for row in ga_summary_rows:
        grouped[str(row["structure"])].append(row)

    vacancy_rows: list[dict[str, object]] = []

    for structure_name in sorted(grouped):
        rows = grouped[structure_name]

        mean_abs_bond_values = [
            float(row["mean_absolute_bond_change_A"])
            for row in rows
            if row["mean_absolute_bond_change_A"] is not None
        ]

        rms_bond_values = [
            float(row["rms_bond_change_A"])
            for row in rows
            if row["rms_bond_change_A"] is not None
        ]

        max_abs_bond_values = [
            float(row["maximum_absolute_bond_change_A"])
            for row in rows
            if row["maximum_absolute_bond_change_A"] is not None
        ]

        mean_abs_angle_values = [
            float(row["mean_absolute_angle_change_deg"])
            for row in rows
            if row["mean_absolute_angle_change_deg"] is not None
        ]

        rms_angle_values = [
            float(row["rms_angle_change_deg"])
            for row in rows
            if row["rms_angle_change_deg"] is not None
        ]

        max_abs_angle_values = [
            float(row["maximum_absolute_angle_change_deg"])
            for row in rows
            if row["maximum_absolute_angle_change_deg"] is not None
        ]

        vacancy_rows.append(
            {
                "structure": structure_name,
                "number_of_first_shell_ga":
                    len(rows),
                "mean_of_ga_mean_absolute_bond_change_A":
                    mean(mean_abs_bond_values),
                "mean_of_ga_rms_bond_change_A":
                    mean(rms_bond_values),
                "maximum_local_bond_change_A":
                    max(max_abs_bond_values),
                "mean_of_ga_mean_absolute_angle_change_deg":
                    mean(mean_abs_angle_values),
                "mean_of_ga_rms_angle_change_deg":
                    mean(rms_angle_values),
                "maximum_local_angle_change_deg":
                    max(max_abs_angle_values),
            }
        )

    return vacancy_rows


# ============================================================
# Main program
# ============================================================

def main() -> None:
    """Generate Ga-level and vacancy-level distortion summaries."""

    bond_rows = read_csv_rows(
        BOND_CHANGE_PATH
    )

    angle_rows = read_csv_rows(
        ANGLE_CHANGE_PATH
    )

    ga_summary_rows = summarize_by_ga(
        bond_rows=bond_rows,
        angle_rows=angle_rows,
    )

    vacancy_summary_rows = summarize_by_vacancy(
        ga_summary_rows
    )

    write_csv(
        output_path=GA_OUTPUT_PATH,
        rows=ga_summary_rows,
    )

    write_csv(
        output_path=VACANCY_OUTPUT_PATH,
        rows=vacancy_summary_rows,
    )

    print("=" * 78)
    print("Local distortion summary")
    print("=" * 78)

    for row in vacancy_summary_rows:
        print()
        print(row["structure"])
        print(
            "  First-shell Ga atoms: "
            f"{row['number_of_first_shell_ga']}"
        )
        print(
            "  Mean |Δr|: "
            f"{row['mean_of_ga_mean_absolute_bond_change_A']:.4f} Å"
        )
        print(
            "  Mean RMS Δr: "
            f"{row['mean_of_ga_rms_bond_change_A']:.4f} Å"
        )
        print(
            "  Mean |Δθ|: "
            f"{row['mean_of_ga_mean_absolute_angle_change_deg']:.4f}°"
        )
        print(
            "  Mean RMS Δθ: "
            f"{row['mean_of_ga_rms_angle_change_deg']:.4f}°"
        )

    print()
    print("=" * 78)
    print("Output files")
    print("=" * 78)
    print(GA_OUTPUT_PATH)
    print(VACANCY_OUTPUT_PATH)


if __name__ == "__main__":
    main()