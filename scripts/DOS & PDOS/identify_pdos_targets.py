#!/usr/bin/env python3
"""
Identify first-shell Ga atoms for site-projected DOS analysis.

Input
-----
analysis/local_structure/local_distortion_by_ga.csv

Output
------
analysis/electronic_structure/dos/pdos_targets.csv

Each output row corresponds to one first-shell Ga atom around an
oxygen vacancy.

Important indexing convention
-----------------------------
The CSV atom numbers are one-based:

    Ga29 = atom number 29

pymatgen uses zero-based Python indices:

    Ga29 -> Python index 28

For PDOS extraction from the vacancy vasprun.xml, use:

    defect_ga_python_index

rather than the pristine/reference atom index.
"""

from __future__ import annotations

import csv
from pathlib import Path


# ============================================================
# Project paths
# ============================================================

def find_project_root() -> Path:
    """
    Find the project root from the location of this script.

    Expected script location:

        project_root/
            scripts/
                DOS & PDOS/
                    identify_pdos_targets.py
    """

    script_path = Path(__file__).resolve()

    for parent in script_path.parents:
        if (
            (parent / "analysis").is_dir()
            and (
                parent
                / "beta-Ga2O3_oxygen_vacancy_VASP_inputs"
            ).is_dir()
        ):
            return parent

    raise RuntimeError(
        "Cannot identify the project root.\n"
        f"Script location:\n{script_path}"
    )


PROJECT_ROOT = find_project_root()

LOCAL_STRUCTURE_ROOT = (
    PROJECT_ROOT
    / "analysis"
    / "local_structure"
)

ELECTRONIC_STRUCTURE_ROOT = (
    PROJECT_ROOT
    / "analysis"
    / "electronic_structure"
)

INPUT_CSV_PATH = (
    LOCAL_STRUCTURE_ROOT
    / "local_distortion_by_ga.csv"
)

OUTPUT_DIR = (
    ELECTRONIC_STRUCTURE_ROOT
    / "dos"
)

OUTPUT_CSV_PATH = (
    OUTPUT_DIR
    / "pdos_targets.csv"
)


# ============================================================
# CSV helpers
# ============================================================

def read_csv_rows(
    file_path: Path,
) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Cannot find input CSV file:\n{file_path}"
        )

    with file_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(
            csv.DictReader(csv_file)
        )

    if not rows:
        raise ValueError(
            f"The input CSV contains no data rows:\n"
            f"{file_path}"
        )

    return rows


def write_csv(
    file_path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write dictionaries to a CSV file."""

    if not rows:
        raise ValueError(
            "No PDOS target rows were generated."
        )

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
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
# Column handling
# ============================================================

def find_column(
    row: dict[str, str],
    possible_names: tuple[str, ...],
    description: str,
    required: bool = True,
) -> str | None:
    """
    Return the first available column among possible names.
    """

    for column_name in possible_names:
        if column_name in row:
            return column_name

    if required:
        available_columns = "\n".join(
            f"  - {column}"
            for column in row.keys()
        )

        attempted_columns = "\n".join(
            f"  - {column}"
            for column in possible_names
        )

        raise KeyError(
            f"Cannot identify the column for {description}.\n\n"
            f"Tried:\n{attempted_columns}\n\n"
            f"Available columns:\n{available_columns}"
        )

    return None


def parse_optional_float(
    value: str | None,
) -> float | None:
    """Convert a non-empty string to float."""

    if value is None:
        return None

    stripped_value = value.strip()

    if stripped_value == "":
        return None

    return float(stripped_value)


# ============================================================
# Coordination handling
# ============================================================

def parse_coordination_change(
    coordination_change: str,
) -> tuple[int, int]:
    """
    Parse a coordination-change string such as:

        4 -> 3
        6 → 5

    Returns
    -------
    pristine_coordination
    vacancy_coordination
    """

    normalized_text = (
        coordination_change
        .replace("→", "->")
        .replace("–>", "->")
        .strip()
    )

    parts = normalized_text.split("->")

    if len(parts) != 2:
        raise ValueError(
            "Cannot parse coordination change:\n"
            f"{coordination_change!r}"
        )

    pristine_coordination = int(
        parts[0].strip()
    )

    vacancy_coordination = int(
        parts[1].strip()
    )

    return (
        pristine_coordination,
        vacancy_coordination,
    )


def classify_ga_site(
    pristine_coordination: int,
) -> str:
    """
    Classify the pristine Ga coordination environment.
    """

    if pristine_coordination == 4:
        return "tetrahedral"

    if pristine_coordination == 6:
        return "octahedral"

    return f"{pristine_coordination}-coordinate"


# ============================================================
# Build PDOS target table
# ============================================================

def build_pdos_target_rows(
    input_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """
    Extract first-shell Ga atoms and prepare PDOS indices.
    """

    example_row = input_rows[0]

    structure_column = find_column(
        example_row,
        (
            "structure",
            "structure_name",
        ),
        "structure name",
    )

    vacancy_label_column = find_column(
        example_row,
        (
            "vacancy_label",
        ),
        "vacancy label",
        required=False,
    )

    first_shell_rank_column = find_column(
        example_row,
        (
            "first_shell_rank",
            "shell_rank",
            "rank",
        ),
        "first-shell rank",
        required=False,
    )

    reference_ga_column = find_column(
        example_row,
        (
            "reference_ga_atom_number",
            "reference_atom_number",
        ),
        "reference Ga atom number",
    )

    defect_ga_column = find_column(
        example_row,
        (
            "defect_ga_atom_number",
            "vacancy_ga_atom_number",
            "defect_atom_number",
        ),
        "defect Ga atom number",
    )

    distance_column = find_column(
        example_row,
        (
            "ga_initial_distance_to_vacancy_A",
            "ga_distance_to_vacancy_A",
            "distance_to_vacancy_A",
        ),
        "Ga distance to vacancy",
    )

    coordination_change_column = find_column(
        example_row,
        (
            "coordination_change",
        ),
        "coordination change",
    )

    mean_absolute_bond_column = find_column(
        example_row,
        (
            "mean_absolute_bond_change_A",
        ),
        "mean absolute bond change",
        required=False,
    )

    maximum_absolute_bond_column = find_column(
        example_row,
        (
            "maximum_absolute_bond_change_A",
        ),
        "maximum absolute bond change",
        required=False,
    )

    mean_absolute_angle_column = find_column(
        example_row,
        (
            "mean_absolute_angle_change_deg",
        ),
        "mean absolute angle change",
        required=False,
    )

    maximum_absolute_angle_column = find_column(
        example_row,
        (
            "maximum_absolute_angle_change_deg",
        ),
        "maximum absolute angle change",
        required=False,
    )

    output_rows: list[
        dict[str, object]
    ] = []

    for input_row in input_rows:
        structure_name = input_row[
            structure_column
        ]

        vacancy_label = (
            input_row[vacancy_label_column]
            if vacancy_label_column is not None
            else structure_name.replace(
                "vacancy_",
                "",
            )
        )

        reference_ga_atom_number = int(
            input_row[
                reference_ga_column
            ]
        )

        defect_ga_atom_number = int(
            input_row[
                defect_ga_column
            ]
        )

        (
            pristine_coordination,
            vacancy_coordination,
        ) = parse_coordination_change(
            input_row[
                coordination_change_column
            ]
        )

        coordination_type = classify_ga_site(
            pristine_coordination
        )

        shell_rank: int | str = ""

        if first_shell_rank_column is not None:
            rank_value = input_row.get(
                first_shell_rank_column,
                "",
            ).strip()

            if rank_value != "":
                shell_rank = int(
                    float(rank_value)
                )

        output_rows.append(
            {
                "structure":
                    structure_name,
                "vacancy_label":
                    vacancy_label,
                "first_shell_rank":
                    shell_rank,

                # One-based atom numbers shown in POSCAR-style output.
                "reference_ga_atom_number":
                    reference_ga_atom_number,
                "defect_ga_atom_number":
                    defect_ga_atom_number,

                # Zero-based pymatgen indices.
                "reference_ga_python_index":
                    reference_ga_atom_number - 1,
                "defect_ga_python_index":
                    defect_ga_atom_number - 1,

                "ga_initial_distance_to_vacancy_A":
                    float(
                        input_row[
                            distance_column
                        ]
                    ),

                "pristine_coordination_number":
                    pristine_coordination,
                "vacancy_coordination_number":
                    vacancy_coordination,
                "coordination_change":
                    (
                        f"{pristine_coordination}"
                        f" -> "
                        f"{vacancy_coordination}"
                    ),
                "coordination_type":
                    coordination_type,

                "mean_absolute_bond_change_A":
                    (
                        parse_optional_float(
                            input_row.get(
                                mean_absolute_bond_column
                            )
                        )
                        if mean_absolute_bond_column
                        is not None
                        else None
                    ),

                "maximum_absolute_bond_change_A":
                    (
                        parse_optional_float(
                            input_row.get(
                                maximum_absolute_bond_column
                            )
                        )
                        if maximum_absolute_bond_column
                        is not None
                        else None
                    ),

                "mean_absolute_angle_change_deg":
                    (
                        parse_optional_float(
                            input_row.get(
                                mean_absolute_angle_column
                            )
                        )
                        if mean_absolute_angle_column
                        is not None
                        else None
                    ),

                "maximum_absolute_angle_change_deg":
                    (
                        parse_optional_float(
                            input_row.get(
                                maximum_absolute_angle_column
                            )
                        )
                        if maximum_absolute_angle_column
                        is not None
                        else None
                    ),

                # This is the index that the future PDOS script should use.
                "pdos_python_index":
                    defect_ga_atom_number - 1,
            }
        )

    vacancy_order = {
        "O1": 1,
        "O2": 2,
        "O3": 3,
    }

    output_rows.sort(
        key=lambda row: (
            vacancy_order.get(
                str(row["vacancy_label"]),
                999,
            ),
            (
                int(row["first_shell_rank"])
                if row["first_shell_rank"] != ""
                else 999
            ),
            float(
                row[
                    "ga_initial_distance_to_vacancy_A"
                ]
            ),
        )
    )

    return output_rows


# ============================================================
# Validation against vasprun structures
# ============================================================

def validate_target_indices(
    target_rows: list[dict[str, object]],
) -> None:
    """
    Perform basic index checks before later PDOS extraction.

    Vacancy structures contain 79 atoms in the current project.
    This function only checks that all indices are non-negative
    and internally consistent.
    """

    for row in target_rows:
        defect_atom_number = int(
            row["defect_ga_atom_number"]
        )

        defect_python_index = int(
            row["defect_ga_python_index"]
        )

        if defect_atom_number < 1:
            raise ValueError(
                "Invalid one-based defect atom number:\n"
                f"{row}"
            )

        if defect_python_index != (
            defect_atom_number - 1
        ):
            raise ValueError(
                "Defect atom number/index conversion failed:\n"
                f"{row}"
            )

        if defect_python_index < 0:
            raise ValueError(
                "Negative PDOS Python index:\n"
                f"{row}"
            )


# ============================================================
# Terminal output
# ============================================================

def print_target_summary(
    target_rows: list[dict[str, object]],
) -> None:
    """Print targets grouped by vacancy structure."""

    print("=" * 78)
    print("PDOS targets: first-shell Ga atoms")
    print("=" * 78)

    current_structure: str | None = None

    for row in target_rows:
        structure_name = str(
            row["structure"]
        )

        if structure_name != current_structure:
            print()
            print(structure_name)
            print("-" * 78)

            current_structure = structure_name

        print(
            f"Ga{int(row['defect_ga_atom_number']):>2d} "
            f"(Python index "
            f"{int(row['defect_ga_python_index']):>2d})"
        )

        print(
            "  Reference Ga          : "
            f"Ga{int(row['reference_ga_atom_number'])}"
        )

        print(
            "  Coordination          : "
            f"{row['coordination_change']} "
            f"({row['coordination_type']})"
        )

        print(
            "  Distance to vacancy   : "
            f"{float(row['ga_initial_distance_to_vacancy_A']):.4f} Å"
        )

        mean_bond_change = row[
            "mean_absolute_bond_change_A"
        ]

        mean_angle_change = row[
            "mean_absolute_angle_change_deg"
        ]

        if mean_bond_change is not None:
            print(
                "  Mean |Δr|             : "
                f"{float(mean_bond_change):.4f} Å"
            )

        if mean_angle_change is not None:
            print(
                "  Mean |Δθ|             : "
                f"{float(mean_angle_change):.4f}°"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Generate the PDOS target table."""

    input_rows = read_csv_rows(
        INPUT_CSV_PATH
    )

    target_rows = build_pdos_target_rows(
        input_rows
    )

    validate_target_indices(
        target_rows
    )

    write_csv(
        OUTPUT_CSV_PATH,
        target_rows,
    )

    print_target_summary(
        target_rows
    )

    print()
    print("=" * 78)
    print("Output file")
    print("=" * 78)
    print(OUTPUT_CSV_PATH)

    print()
    print(
        "Use the 'pdos_python_index' column when extracting "
        "site-projected DOS from each vacancy vasprun.xml."
    )


if __name__ == "__main__":
    main()