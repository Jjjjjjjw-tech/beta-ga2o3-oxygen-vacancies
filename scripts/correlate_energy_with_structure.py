#!/usr/bin/env python3
"""
Correlate oxygen-vacancy relative energies with structural descriptors.

Inputs
------
analysis/energy/vacancy_relative_energies.csv
analysis/local_structure/structure_descriptors.csv

Outputs
-------
analysis/correlation/energy_structure_summary.csv

analysis/correlation/figures/
    energy_vs_mean_ga_displacement.png
    energy_vs_maximum_ga_displacement.png
    energy_vs_mean_bond_distortion.png
    energy_vs_maximum_bond_distortion.png
    energy_vs_mean_angle_distortion.png
    energy_vs_maximum_angle_distortion.png
    energy_vs_first_shell_ga.png

Important
---------
Only three vacancy configurations are available. The plots are therefore
used for descriptive trend comparison, not for statistically robust
correlation analysis.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# File paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENERGY_INPUT_PATH = (
    PROJECT_ROOT
    / "analysis"
    / "energy"
    / "vacancy_relative_energies.csv"
)

DESCRIPTOR_INPUT_PATH = (
    PROJECT_ROOT
    / "analysis"
    / "local_structure"
    / "structure_descriptors.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "analysis"
    / "correlation"
)

FIGURE_DIR = (
    OUTPUT_ROOT
    / "figures"
)

SUMMARY_OUTPUT_PATH = (
    OUTPUT_ROOT
    / "energy_structure_summary.csv"
)


# ============================================================
# Plot configuration
# ============================================================

PLOT_DESCRIPTORS = (
    {
        "column": "mean_ga_displacement_A",
        "x_label": "Mean Ga displacement (Å)",
        "title": "Relative energy vs mean Ga displacement",
        "filename": "energy_vs_mean_ga_displacement.png",
    },
    {
        "column": "maximum_ga_displacement_A",
        "x_label": "Maximum Ga displacement (Å)",
        "title": "Relative energy vs maximum Ga displacement",
        "filename": "energy_vs_maximum_ga_displacement.png",
    },
    {
        "column": "mean_absolute_bond_distortion_A",
        "x_label": "Mean absolute Ga–O bond distortion (Å)",
        "title": "Relative energy vs mean bond distortion",
        "filename": "energy_vs_mean_bond_distortion.png",
    },
    {
        "column": "maximum_local_bond_distortion_A",
        "x_label": "Maximum local Ga–O bond distortion (Å)",
        "title": "Relative energy vs maximum bond distortion",
        "filename": "energy_vs_maximum_bond_distortion.png",
    },
    {
        "column": "mean_absolute_angle_distortion_deg",
        "x_label": "Mean absolute O–Ga–O angle distortion (°)",
        "title": "Relative energy vs mean angle distortion",
        "filename": "energy_vs_mean_angle_distortion.png",
    },
    {
        "column": "maximum_local_angle_distortion_deg",
        "x_label": "Maximum local O–Ga–O angle distortion (°)",
        "title": "Relative energy vs maximum angle distortion",
        "filename": "energy_vs_maximum_angle_distortion.png",
    },
    {
        "column": "number_of_first_shell_ga",
        "x_label": "Number of first-shell Ga atoms",
        "title": "Relative energy vs first-shell Ga number",
        "filename": "energy_vs_first_shell_ga.png",
    },
)


# ============================================================
# CSV helper functions
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


# ============================================================
# Merge energy and descriptor tables
# ============================================================

def build_energy_dictionary(
    energy_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Map each vacancy structure name to its energy row."""

    energy_dictionary: dict[
        str,
        dict[str, str],
    ] = {}

    for row in energy_rows:
        structure_name = row["structure"]

        if structure_name in energy_dictionary:
            raise ValueError(
                f"Duplicate energy row found for "
                f"{structure_name}."
            )

        energy_dictionary[
            structure_name
        ] = row

    return energy_dictionary


def merge_energy_and_descriptors(
    energy_rows: list[dict[str, str]],
    descriptor_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """
    Merge relative-energy and structural-descriptor data.

    The structure column is used as the matching key.
    """

    energy_dictionary = (
        build_energy_dictionary(
            energy_rows
        )
    )

    merged_rows: list[
        dict[str, object]
    ] = []

    for descriptor_row in descriptor_rows:
        structure_name = descriptor_row[
            "structure"
        ]

        if structure_name not in energy_dictionary:
            raise KeyError(
                f"No relative-energy row was found for "
                f"{structure_name}."
            )

        energy_row = energy_dictionary[
            structure_name
        ]

        merged_row: dict[
            str,
            object
        ] = {
            "structure": structure_name,
            "vacancy_label":
                descriptor_row["vacancy_label"],
            "energy_rank":
                int(energy_row["rank"]),
            "total_energy_eV":
                float(
                    energy_row[
                        "total_energy_eV"
                    ]
                ),
            "relative_energy_eV":
                float(
                    energy_row[
                        "relative_energy_eV"
                    ]
                ),
        }

        for column_name, value in (
            descriptor_row.items()
        ):
            if column_name in {
                "structure",
                "vacancy_label",
            }:
                continue

            merged_row[
                column_name
            ] = value

        merged_rows.append(
            merged_row
        )

    merged_rows.sort(
        key=lambda row: int(
            row["energy_rank"]
        )
    )

    return merged_rows


# ============================================================
# Plot helper functions
# ============================================================

def validate_plot_columns(
    rows: list[dict[str, object]],
) -> None:
    """Check that all configured descriptor columns exist."""

    available_columns = set(
        rows[0].keys()
    )

    missing_columns = {
        str(config["column"])
        for config in PLOT_DESCRIPTORS
        if config["column"]
        not in available_columns
    }

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        available_text = ", ".join(
            sorted(available_columns)
        )

        raise KeyError(
            "The merged table is missing configured "
            "descriptor columns:\n"
            f"{missing_text}\n\n"
            "Available columns:\n"
            f"{available_text}"
        )


def plot_energy_vs_descriptor(
    rows: list[dict[str, object]],
    descriptor_column: str,
    x_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot relative energy against one structural descriptor."""

    x_values = [
        float(row[descriptor_column])
        for row in rows
    ]

    y_values = [
        float(row["relative_energy_eV"])
        for row in rows
    ]

    vacancy_labels = [
        str(row["vacancy_label"])
        for row in rows
    ]

    figure, axis = plt.subplots(
        figsize=(6.5, 5),
    )

    axis.scatter(
        x_values,
        y_values,
        s=80,
    )

    for x_value, y_value, label in zip(
        x_values,
        y_values,
        vacancy_labels,
    ):
        axis.annotate(
            label,
            xy=(
                x_value,
                y_value,
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
        )

    axis.set_xlabel(
        x_label
    )

    axis.set_ylabel(
        "Relative vacancy energy (eV)"
    )

    axis.set_title(
        title
    )

    axis.set_ylim(
        bottom=min(
            0.0,
            min(y_values) - 0.03,
        )
    )

    axis.grid(
        alpha=0.25
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Terminal summary
# ============================================================

def print_summary(
    rows: list[dict[str, object]],
) -> None:
    """Print the merged energy–structure table."""

    print("=" * 78)
    print("Energy–structure summary")
    print("=" * 78)

    for row in rows:
        print()
        print(
            f"{row['structure']} "
            f"(rank {row['energy_rank']})"
        )

        print(
            "  Relative energy: "
            f"{float(row['relative_energy_eV']):.6f} eV"
        )

        print(
            "  First-shell Ga: "
            f"{int(row['number_of_first_shell_ga'])}"
        )

        print(
            "  Mean Ga displacement: "
            f"{float(row['mean_ga_displacement_A']):.4f} Å"
        )

        print(
            "  Maximum Ga displacement: "
            f"{float(row['maximum_ga_displacement_A']):.4f} Å"
        )

        print(
            "  Mean |Δr|: "
            f"{float(row['mean_absolute_bond_distortion_A']):.4f} Å"
        )

        print(
            "  Maximum |Δr|: "
            f"{float(row['maximum_local_bond_distortion_A']):.4f} Å"
        )

        print(
            "  Mean |Δθ|: "
            f"{float(row['mean_absolute_angle_distortion_deg']):.4f}°"
        )

        print(
            "  Maximum |Δθ|: "
            f"{float(row['maximum_local_angle_distortion_deg']):.4f}°"
        )


# ============================================================
# Main program
# ============================================================

def main() -> None:
    """Merge energy and structure descriptors and generate plots."""

    energy_rows = read_csv_rows(
        ENERGY_INPUT_PATH
    )

    descriptor_rows = read_csv_rows(
        DESCRIPTOR_INPUT_PATH
    )

    merged_rows = (
        merge_energy_and_descriptors(
            energy_rows=energy_rows,
            descriptor_rows=
                descriptor_rows,
        )
    )

    validate_plot_columns(
        merged_rows
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output_path=
            SUMMARY_OUTPUT_PATH,
        rows=merged_rows,
    )

    for config in PLOT_DESCRIPTORS:
        output_path = (
            FIGURE_DIR
            / str(config["filename"])
        )

        plot_energy_vs_descriptor(
            rows=merged_rows,
            descriptor_column=str(
                config["column"]
            ),
            x_label=str(
                config["x_label"]
            ),
            title=str(
                config["title"]
            ),
            output_path=output_path,
        )

    print_summary(
        merged_rows
    )

    print()
    print("=" * 78)
    print("Output files")
    print("=" * 78)

    print(SUMMARY_OUTPUT_PATH)

    for config in PLOT_DESCRIPTORS:
        print(
            FIGURE_DIR
            / str(config["filename"])
        )

    print()
    print(
        "Note: only three vacancy configurations are "
        "available. These figures show descriptive trends "
        "and should not be interpreted as statistically "
        "robust correlations."
    )


if __name__ == "__main__":
    main()