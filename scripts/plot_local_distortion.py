#!/usr/bin/env python3
"""
Plot vacancy-level local structural distortion with error bars.

Input
-----
analysis/local_structure/local_distortion_by_ga.csv

Outputs
-------
analysis/local_structure/figures/
    local_bond_distortion_with_error_bars.png
    local_angle_distortion_with_error_bars.png

analysis/local_structure/
    local_distortion_plot_statistics.csv

Interpretation
--------------
For each vacancy:

- Bar height:
  mean value across all first-shell Ga atoms.

- Error bar:
  population standard deviation across the first-shell Ga atoms.

The error bars therefore describe variation among neighboring Ga atoms.
They do not represent numerical uncertainty in the VASP calculation.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# File paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCAL_STRUCTURE_ROOT = (
    PROJECT_ROOT
    / "analysis"
    / "local_structure"
)

INPUT_CSV_PATH = (
    LOCAL_STRUCTURE_ROOT
    / "local_distortion_by_ga.csv"
)

FIGURE_DIR = (
    LOCAL_STRUCTURE_ROOT
    / "figures"
)

BOND_FIGURE_PATH = (
    FIGURE_DIR
    / "local_bond_distortion_with_error_bars.png"
)

ANGLE_FIGURE_PATH = (
    FIGURE_DIR
    / "local_angle_distortion_with_error_bars.png"
)

STATISTICS_OUTPUT_PATH = (
    LOCAL_STRUCTURE_ROOT
    / "local_distortion_plot_statistics.csv"
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
            f"Cannot find input CSV file:\n{file_path}"
        )

    with file_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(
            f"The input CSV contains no data rows:\n{file_path}"
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
# Data preparation
# ============================================================

def format_vacancy_label(
    structure_name: str,
) -> str:
    """Convert vacancy_O1 to O1."""

    return structure_name.replace(
        "vacancy_",
        "",
    )


def population_standard_deviation(
    values: list[float],
) -> float:
    """
    Calculate population standard deviation.

    If only one value is present, the spread is defined as zero.
    """

    if not values:
        raise ValueError(
            "Cannot calculate standard deviation "
            "for an empty list."
        )

    if len(values) == 1:
        return 0.0

    return statistics.pstdev(values)


def mean(values: list[float]) -> float:
    """Calculate arithmetic mean."""

    if not values:
        raise ValueError(
            "Cannot calculate mean for an empty list."
        )

    return statistics.fmean(values)


def prepare_vacancy_statistics(
    rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """
    Calculate vacancy-level means and standard deviations.

    Each input row corresponds to one first-shell Ga atom.
    """

    required_columns = {
        "structure",
        "mean_absolute_bond_change_A",
        "rms_bond_change_A",
        "mean_absolute_angle_change_deg",
        "rms_angle_change_deg",
    }

    missing_columns = required_columns - set(
        rows[0].keys()
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise KeyError(
            "The input CSV is missing required columns:\n"
            f"{missing_text}"
        )

    grouped_rows: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        grouped_rows[
            row["structure"]
        ].append(row)

    statistics_rows: list[dict[str, object]] = []

    for structure_name in sorted(grouped_rows):
        vacancy_rows = grouped_rows[
            structure_name
        ]

        mean_abs_bond_values = [
            float(
                row[
                    "mean_absolute_bond_change_A"
                ]
            )
            for row in vacancy_rows
        ]

        rms_bond_values = [
            float(
                row["rms_bond_change_A"]
            )
            for row in vacancy_rows
        ]

        mean_abs_angle_values = [
            float(
                row[
                    "mean_absolute_angle_change_deg"
                ]
            )
            for row in vacancy_rows
        ]

        rms_angle_values = [
            float(
                row["rms_angle_change_deg"]
            )
            for row in vacancy_rows
        ]

        statistics_rows.append(
            {
                "structure": structure_name,
                "vacancy_label":
                    format_vacancy_label(
                        structure_name
                    ),
                "number_of_first_shell_ga":
                    len(vacancy_rows),

                "mean_absolute_bond_change_A":
                    mean(mean_abs_bond_values),
                "std_absolute_bond_change_A":
                    population_standard_deviation(
                        mean_abs_bond_values
                    ),

                "mean_rms_bond_change_A":
                    mean(rms_bond_values),
                "std_rms_bond_change_A":
                    population_standard_deviation(
                        rms_bond_values
                    ),

                "mean_absolute_angle_change_deg":
                    mean(mean_abs_angle_values),
                "std_absolute_angle_change_deg":
                    population_standard_deviation(
                        mean_abs_angle_values
                    ),

                "mean_rms_angle_change_deg":
                    mean(rms_angle_values),
                "std_rms_angle_change_deg":
                    population_standard_deviation(
                        rms_angle_values
                    ),
            }
        )

    return statistics_rows


# ============================================================
# Figure helper functions
# ============================================================

def add_value_labels(
    axis,
    bars,
    errors: list[float],
    decimal_places: int,
) -> None:
    """
    Add values above the upper end of each error bar.
    """

    for bar, error in zip(
        bars,
        errors,
    ):
        height = bar.get_height()

        axis.annotate(
            f"{height:.{decimal_places}f}",
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                height + error,
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def set_upper_limit(
    axis,
    values: list[float],
    errors: list[float],
) -> None:
    """Leave space above bars and annotations."""

    upper_values = [
        value + error
        for value, error in zip(
            values,
            errors,
        )
    ]

    maximum = max(upper_values)

    if maximum > 0:
        axis.set_ylim(
            0,
            maximum * 1.25,
        )
    else:
        axis.set_ylim(
            bottom=0,
        )


# ============================================================
# Bond-length distortion figure
# ============================================================

def plot_bond_distortion(
    statistics_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """
    Plot bond-length distortion with Ga-to-Ga error bars.
    """

    vacancy_labels = [
        str(row["vacancy_label"])
        for row in statistics_rows
    ]

    mean_absolute_values = [
        float(
            row[
                "mean_absolute_bond_change_A"
            ]
        )
        for row in statistics_rows
    ]

    mean_absolute_errors = [
        float(
            row[
                "std_absolute_bond_change_A"
            ]
        )
        for row in statistics_rows
    ]

    rms_values = [
        float(
            row[
                "mean_rms_bond_change_A"
            ]
        )
        for row in statistics_rows
    ]

    rms_errors = [
        float(
            row[
                "std_rms_bond_change_A"
            ]
        )
        for row in statistics_rows
    ]

    x_positions = list(
        range(len(vacancy_labels))
    )

    bar_width = 0.34

    left_positions = [
        position - bar_width / 2
        for position in x_positions
    ]

    right_positions = [
        position + bar_width / 2
        for position in x_positions
    ]

    figure, axis = plt.subplots(
        figsize=(7, 5),
    )

    mean_bars = axis.bar(
        left_positions,
        mean_absolute_values,
        width=bar_width,
        yerr=mean_absolute_errors,
        capsize=5,
        label="Mean |Δr|",
    )

    rms_bars = axis.bar(
        right_positions,
        rms_values,
        width=bar_width,
        yerr=rms_errors,
        capsize=5,
        label="Mean RMS Δr",
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        vacancy_labels
    )

    axis.set_xlabel(
        "Oxygen vacancy site"
    )

    axis.set_ylabel(
        "Ga–O bond-length distortion (Å)"
    )

    axis.set_title(
        "Local Ga–O bond-length distortion"
    )

    axis.legend(
        frameon=False
    )

    all_values = (
        mean_absolute_values
        + rms_values
    )

    all_errors = (
        mean_absolute_errors
        + rms_errors
    )

    set_upper_limit(
        axis=axis,
        values=all_values,
        errors=all_errors,
    )

    add_value_labels(
        axis=axis,
        bars=mean_bars,
        errors=mean_absolute_errors,
        decimal_places=4,
    )

    add_value_labels(
        axis=axis,
        bars=rms_bars,
        errors=rms_errors,
        decimal_places=4,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Bond-angle distortion figure
# ============================================================

def plot_angle_distortion(
    statistics_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """
    Plot bond-angle distortion with Ga-to-Ga error bars.
    """

    vacancy_labels = [
        str(row["vacancy_label"])
        for row in statistics_rows
    ]

    mean_absolute_values = [
        float(
            row[
                "mean_absolute_angle_change_deg"
            ]
        )
        for row in statistics_rows
    ]

    mean_absolute_errors = [
        float(
            row[
                "std_absolute_angle_change_deg"
            ]
        )
        for row in statistics_rows
    ]

    rms_values = [
        float(
            row[
                "mean_rms_angle_change_deg"
            ]
        )
        for row in statistics_rows
    ]

    rms_errors = [
        float(
            row[
                "std_rms_angle_change_deg"
            ]
        )
        for row in statistics_rows
    ]

    x_positions = list(
        range(len(vacancy_labels))
    )

    bar_width = 0.34

    left_positions = [
        position - bar_width / 2
        for position in x_positions
    ]

    right_positions = [
        position + bar_width / 2
        for position in x_positions
    ]

    figure, axis = plt.subplots(
        figsize=(7, 5),
    )

    mean_bars = axis.bar(
        left_positions,
        mean_absolute_values,
        width=bar_width,
        yerr=mean_absolute_errors,
        capsize=5,
        label="Mean |Δθ|",
    )

    rms_bars = axis.bar(
        right_positions,
        rms_values,
        width=bar_width,
        yerr=rms_errors,
        capsize=5,
        label="Mean RMS Δθ",
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        vacancy_labels
    )

    axis.set_xlabel(
        "Oxygen vacancy site"
    )

    axis.set_ylabel(
        "O–Ga–O bond-angle distortion (°)"
    )

    axis.set_title(
        "Local O–Ga–O bond-angle distortion"
    )

    axis.legend(
        frameon=False
    )

    all_values = (
        mean_absolute_values
        + rms_values
    )

    all_errors = (
        mean_absolute_errors
        + rms_errors
    )

    set_upper_limit(
        axis=axis,
        values=all_values,
        errors=all_errors,
    )

    add_value_labels(
        axis=axis,
        bars=mean_bars,
        errors=mean_absolute_errors,
        decimal_places=3,
    )

    add_value_labels(
        axis=axis,
        bars=rms_bars,
        errors=rms_errors,
        decimal_places=3,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Main program
# ============================================================

def main() -> None:
    """
    Generate local-distortion figures with error bars.
    """

    ga_rows = read_csv_rows(
        INPUT_CSV_PATH
    )

    statistics_rows = (
        prepare_vacancy_statistics(
            ga_rows
        )
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output_path=STATISTICS_OUTPUT_PATH,
        rows=statistics_rows,
    )

    plot_bond_distortion(
        statistics_rows=statistics_rows,
        output_path=BOND_FIGURE_PATH,
    )

    plot_angle_distortion(
        statistics_rows=statistics_rows,
        output_path=ANGLE_FIGURE_PATH,
    )

    print("=" * 78)
    print("Local distortion figures with error bars")
    print("=" * 78)

    for row in statistics_rows:
        print()
        print(row["structure"])

        print(
            "  Number of first-shell Ga: "
            f"{row['number_of_first_shell_ga']}"
        )

        print(
            "  Mean |Δr|: "
            f"{row['mean_absolute_bond_change_A']:.4f} "
            "± "
            f"{row['std_absolute_bond_change_A']:.4f} Å"
        )

        print(
            "  Mean |Δθ|: "
            f"{row['mean_absolute_angle_change_deg']:.4f} "
            "± "
            f"{row['std_absolute_angle_change_deg']:.4f}°"
        )

    print()
    print("=" * 78)
    print("Output files")
    print("=" * 78)

    print(BOND_FIGURE_PATH)
    print(ANGLE_FIGURE_PATH)
    print(STATISTICS_OUTPUT_PATH)


if __name__ == "__main__":
    main()