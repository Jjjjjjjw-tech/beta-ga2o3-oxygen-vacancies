 #!/usr/bin/env python3
"""
Plot a normalized energy–structure summary for oxygen vacancies.

Input
-----
analysis/correlation/energy_structure_summary.csv

Outputs
-------
analysis/correlation/energy_structure_normalized.csv

analysis/correlation/figures/
    energy_structure_grouped_bar.png
    energy_structure_heatmap.png

Notes
-----
Each descriptor is normalized independently with min–max scaling:

    normalized = (value - minimum) / (maximum - minimum)

The normalized values only compare O1, O2, and O3 within the same
descriptor. They do not combine eV, Å, and degrees into one physical
quantity.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# File paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORRELATION_ROOT = (
    PROJECT_ROOT
    / "analysis"
    / "correlation"
)

INPUT_CSV_PATH = (
    CORRELATION_ROOT
    / "energy_structure_summary.csv"
)

NORMALIZED_OUTPUT_PATH = (
    CORRELATION_ROOT
    / "energy_structure_normalized.csv"
)

FIGURE_DIR = (
    CORRELATION_ROOT
    / "figures"
)

GROUPED_BAR_OUTPUT_PATH = (
    FIGURE_DIR
    / "energy_structure_grouped_bar.png"
)

HEATMAP_OUTPUT_PATH = (
    FIGURE_DIR
    / "energy_structure_heatmap.png"
)


# ============================================================
# Descriptor configuration
# ============================================================

DESCRIPTORS = (
    {
        "column": "relative_energy_eV",
        "label": "Relative energy",
    },
    {
        "column": "maximum_ga_displacement_A",
        "label": "Maximum Ga displacement",
    },
    {
        "column": "mean_absolute_bond_distortion_A",
        "label": "Mean bond distortion",
    },
    {
        "column": "maximum_local_angle_distortion_deg",
        "label": "Maximum angle distortion",
    },
)


# ============================================================
# CSV helpers
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
            f"No data rows available for {output_path.name}."
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
# Validation and sorting
# ============================================================

def validate_columns(
    rows: list[dict[str, str]],
) -> None:
    """Check that all required columns exist."""

    required_columns = {
        "structure",
        "vacancy_label",
    }

    required_columns.update(
        descriptor["column"]
        for descriptor in DESCRIPTORS
    )

    available_columns = set(
        rows[0].keys()
    )

    missing_columns = (
        required_columns
        - available_columns
    )

    if missing_columns:
        raise KeyError(
            "Missing required columns:\n"
            + ", ".join(
                sorted(missing_columns)
            )
            + "\n\nAvailable columns:\n"
            + ", ".join(
                sorted(available_columns)
            )
        )


def sort_rows_by_vacancy(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Sort rows as O1, O2, O3."""

    vacancy_order = {
        "O1": 1,
        "O2": 2,
        "O3": 3,
    }

    return sorted(
        rows,
        key=lambda row: vacancy_order.get(
            row["vacancy_label"],
            999,
        ),
    )


# ============================================================
# Normalization
# ============================================================

def min_max_normalize(
    values: list[float],
) -> list[float]:
    """Normalize values to the range 0–1."""

    if not values:
        raise ValueError(
            "Cannot normalize an empty list."
        )

    minimum = min(values)
    maximum = max(values)

    value_range = maximum - minimum

    if value_range == 0:
        return [
            0.0
            for _ in values
        ]

    return [
        (value - minimum) / value_range
        for value in values
    ]


def prepare_normalized_data(
    rows: list[dict[str, str]],
) -> tuple[
    list[str],
    list[str],
    np.ndarray,
    list[dict[str, object]],
]:
    """
    Prepare labels, normalized matrix, and normalized CSV rows.

    Matrix shape:
        number of vacancies × number of descriptors
    """

    sorted_rows = sort_rows_by_vacancy(
        rows
    )

    vacancy_labels = [
        row["vacancy_label"]
        for row in sorted_rows
    ]

    descriptor_labels = [
        descriptor["label"]
        for descriptor in DESCRIPTORS
    ]

    raw_data: dict[
        str,
        list[float],
    ] = {}

    normalized_data: dict[
        str,
        list[float],
    ] = {}

    for descriptor in DESCRIPTORS:
        column_name = descriptor[
            "column"
        ]

        values = [
            float(row[column_name])
            for row in sorted_rows
        ]

        raw_data[column_name] = values

        normalized_data[column_name] = (
            min_max_normalize(
                values
            )
        )

    normalized_matrix = np.array(
        [
            [
                normalized_data[
                    descriptor["column"]
                ][row_index]
                for descriptor in DESCRIPTORS
            ]
            for row_index in range(
                len(sorted_rows)
            )
        ],
        dtype=float,
    )

    output_rows: list[
        dict[str, object]
    ] = []

    for row_index, row in enumerate(
        sorted_rows
    ):
        output_row: dict[
            str,
            object
        ] = {
            "structure": row["structure"],
            "vacancy_label":
                row["vacancy_label"],
        }

        for descriptor in DESCRIPTORS:
            column_name = descriptor[
                "column"
            ]

            output_row[
                column_name
            ] = raw_data[
                column_name
            ][row_index]

            output_row[
                f"{column_name}_normalized"
            ] = normalized_data[
                column_name
            ][row_index]

        output_rows.append(
            output_row
        )

    return (
        vacancy_labels,
        descriptor_labels,
        normalized_matrix,
        output_rows,
    )


# ============================================================
# Grouped bar chart
# ============================================================

def add_bar_labels(
    axis,
    bars,
) -> None:
    """Add normalized values above bars."""

    for bar in bars:
        height = bar.get_height()

        axis.annotate(
            f"{height:.2f}",
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                height,
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_grouped_bar(
    vacancy_labels: list[str],
    descriptor_labels: list[str],
    normalized_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """Plot grouped normalized bars."""

    number_of_vacancies = len(
        vacancy_labels
    )

    number_of_descriptors = len(
        descriptor_labels
    )

    x_positions = np.arange(
        number_of_vacancies
    )

    total_group_width = 0.80

    bar_width = (
        total_group_width
        / number_of_descriptors
    )

    figure, axis = plt.subplots(
        figsize=(9, 5.5),
    )

    for descriptor_index, descriptor_label in enumerate(
        descriptor_labels
    ):
        offset = (
            descriptor_index
            - (number_of_descriptors - 1) / 2
        ) * bar_width

        positions = (
            x_positions + offset
        )

        bars = axis.bar(
            positions,
            normalized_matrix[
                :,
                descriptor_index,
            ],
            width=bar_width,
            label=descriptor_label,
        )

        add_bar_labels(
            axis=axis,
            bars=bars,
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
        "Normalized value"
    )

    axis.set_title(
        "Energy–structure comparison of oxygen vacancies"
    )

    axis.set_ylim(
        0,
        1.16,
    )

    axis.legend(
        frameon=False,
        ncol=2,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Heatmap
# ============================================================

def plot_heatmap(
    vacancy_labels: list[str],
    descriptor_labels: list[str],
    normalized_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """Plot normalized data as a heatmap."""

    figure, axis = plt.subplots(
        figsize=(9, 4.5),
    )

    image = axis.imshow(
        normalized_matrix,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    axis.set_xticks(
        np.arange(
            len(descriptor_labels)
        )
    )

    axis.set_xticklabels(
        descriptor_labels,
        rotation=25,
        ha="right",
    )

    axis.set_yticks(
        np.arange(
            len(vacancy_labels)
        )
    )

    axis.set_yticklabels(
        vacancy_labels
    )

    axis.set_xlabel(
        "Energy and structural descriptor"
    )

    axis.set_ylabel(
        "Oxygen vacancy site"
    )

    axis.set_title(
        "Normalized energy–structure fingerprint"
    )

    for row_index in range(
        normalized_matrix.shape[0]
    ):
        for column_index in range(
            normalized_matrix.shape[1]
        ):
            value = normalized_matrix[
                row_index,
                column_index,
            ]

            text_color = (
                "white"
                if value >= 0.55
                else "black"
            )

            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Normalized value"
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
    vacancy_labels: list[str],
    descriptor_labels: list[str],
    normalized_matrix: np.ndarray,
) -> None:
    """Print the normalized values."""

    print("=" * 78)
    print("Normalized energy–structure summary")
    print("=" * 78)

    for row_index, vacancy_label in enumerate(
        vacancy_labels
    ):
        print()
        print(vacancy_label)

        for column_index, descriptor_label in enumerate(
            descriptor_labels
        ):
            print(
                f"  {descriptor_label:28s}: "
                f"{normalized_matrix[row_index, column_index]:.4f}"
            )


# ============================================================
# Main program
# ============================================================

def main() -> None:
    """Generate normalized energy–structure summary figures."""

    rows = read_csv_rows(
        INPUT_CSV_PATH
    )

    validate_columns(
        rows
    )

    (
        vacancy_labels,
        descriptor_labels,
        normalized_matrix,
        normalized_output_rows,
    ) = prepare_normalized_data(
        rows
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output_path=
            NORMALIZED_OUTPUT_PATH,
        rows=normalized_output_rows,
    )

    plot_grouped_bar(
        vacancy_labels=
            vacancy_labels,
        descriptor_labels=
            descriptor_labels,
        normalized_matrix=
            normalized_matrix,
        output_path=
            GROUPED_BAR_OUTPUT_PATH,
    )

    plot_heatmap(
        vacancy_labels=
            vacancy_labels,
        descriptor_labels=
            descriptor_labels,
        normalized_matrix=
            normalized_matrix,
        output_path=
            HEATMAP_OUTPUT_PATH,
    )

    print_summary(
        vacancy_labels=
            vacancy_labels,
        descriptor_labels=
            descriptor_labels,
        normalized_matrix=
            normalized_matrix,
    )

    print()
    print("=" * 78)
    print("Output files")
    print("=" * 78)

    print(NORMALIZED_OUTPUT_PATH)
    print(GROUPED_BAR_OUTPUT_PATH)
    print(HEATMAP_OUTPUT_PATH)

    print()
    print(
        "Each descriptor was normalized independently. "
        "The figures compare relative rankings among O1, O2, "
        "and O3, rather than defining a combined physical "
        "distortion quantity."
    )


if __name__ == "__main__":
    main()