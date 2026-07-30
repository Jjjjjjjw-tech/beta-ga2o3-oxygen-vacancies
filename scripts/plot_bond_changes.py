#!/usr/bin/env python3
"""Plot mean Ga–O bond-length changes around oxygen vacancies."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


# ======================================
# File paths
# ======================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "analysis"
    / "local_structure"
    / "ga_bond_change_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "analysis"
    / "local_structure"
    / "figures"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "mean_ga_o_bond_change.png"
)


# ======================================
# Read data
# ======================================

if not INPUT_CSV.is_file():
    raise FileNotFoundError(
        f"Cannot find input CSV:\n{INPUT_CSV}"
    )

with INPUT_CSV.open(
    "r",
    newline="",
    encoding="utf-8",
) as csv_file:
    rows = list(csv.DictReader(csv_file))


# ======================================
# Prepare plotting data
# ======================================

labels = []
mean_changes = []

for row in rows:
    vacancy_name = row["structure"].replace(
        "vacancy_",
        "",
    )

    ga_number = row["reference_ga_atom_number"]

    mean_change = float(
        row["mean_preserved_bond_change_A"]
    )

    labels.append(
        f"{vacancy_name}\nGa{ga_number}"
    )

    mean_changes.append(mean_change)


# ======================================
# Plot
# ======================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

figure, axis = plt.subplots(
    figsize=(10, 5),
)

axis.bar(
    labels,
    mean_changes,
)

axis.axhline(
    y=0,
    linewidth=1,
)

axis.set_xlabel(
    "Vacancy site and first-shell Ga atom"
)

axis.set_ylabel(
    "Mean Ga–O bond-length change (Å)"
)

axis.set_title(
    "Mean Ga–O bond-length changes around oxygen vacancies"
)

axis.tick_params(
    axis="x",
    rotation=0,
)

figure.tight_layout()

figure.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

print(f"Figure saved to:\n{OUTPUT_PATH}")

plt.show()