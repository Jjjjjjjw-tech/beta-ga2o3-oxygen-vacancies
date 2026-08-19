#!/usr/bin/env python3
"""
Analyze site- and orbital-projected DOS of first-shell Ga atoms.

Inputs
------
analysis/electronic_structure/pdos/pdos_targets.csv

beta-Ga2O3_oxygen_vacancy_VASP_inputs/
    received_results/
        2026-08-17_results/
            1/scf/vasprun.xml
            2/scf/vasprun.xml
            3/scf/vasprun.xml
            pristine/scf/vasprun.xml

Outputs
-------
analysis/electronic_structure/pdos/ga/
    vacancy_O1_ga_pdos.csv
    vacancy_O2_ga_pdos.csv
    vacancy_O3_ga_pdos.csv

analysis/electronic_structure/figures/
    vacancy_O1_ga_pdos.png
    vacancy_O2_ga_pdos.png
    vacancy_O3_ga_pdos.png

Method
------
1. Read first-shell Ga atom indices from pdos_targets.csv.
2. Group Ga atoms by their pristine coordination type:
       tetrahedral
       octahedral
3. Extract site-resolved s, p, and d projected DOS.
4. Average the PDOS over all Ga atoms in each coordination group.
5. Shift each vacancy energy axis independently so that E_F = 0 eV.

Important
---------
This first script compares tetrahedral and octahedral Ga atoms inside
each vacancy structure. It does not yet calculate vacancy-minus-pristine
PDOS differences for the same physical Ga atoms.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pymatgen.electronic_structure.core import (
    OrbitalType,
    Spin,
)
from pymatgen.io.vasp.outputs import Vasprun


# ============================================================
# Configuration
# ============================================================

RESULT_DATE = "2026-08-17_results"

VACANCY_STRUCTURES = (
    "vacancy_O1",
    "vacancy_O2",
    "vacancy_O3",
)

# Physical folder names in the newly imported VASP dataset
VACANCY_FOLDER_MAP = {
    "vacancy_O1": "1",
    "vacancy_O2": "2",
    "vacancy_O3": "3",
}

VACANCY_LABELS = {
    "vacancy_O1": "O1 vacancy",
    "vacancy_O2": "O2 vacancy",
    "vacancy_O3": "O3 vacancy",
}

COORDINATION_TYPES = (
    "tetrahedral",
    "octahedral",
)

ORBITALS = (
    OrbitalType.s,
    OrbitalType.p,
    OrbitalType.d,
)

ORBITAL_LABELS = {
    OrbitalType.s: "s",
    OrbitalType.p: "p",
    OrbitalType.d: "d",
}

# Energy window relative to each structure's own Fermi level
ENERGY_MIN_EV = -6.0
ENERGY_MAX_EV = 6.0

NORMALIZE_BY_NUMBER_OF_SITES = True


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_ROOT = (
    PROJECT_ROOT
    / "beta-Ga2O3_oxygen_vacancy_VASP_inputs"
    / "received_results"
    / RESULT_DATE
)

ELECTRONIC_STRUCTURE_DIR = (
    PROJECT_ROOT
    / "analysis"
    / "electronic_structure"
)

# ------------------------------------------------------------
# DOS output directory
# ------------------------------------------------------------

DOS_ROOT = (
    ELECTRONIC_STRUCTURE_DIR
    / "dos"
)

# ------------------------------------------------------------
# PDOS output directory
# ------------------------------------------------------------

PDOS_ROOT = (
    ELECTRONIC_STRUCTURE_DIR
    / "pdos"
)

TARGETS_CSV_PATH = (
    PDOS_ROOT
    / "pdos_targets.csv"
)

PDOS_OUTPUT_DIR = (
    PDOS_ROOT
    / "ga"
)

# ------------------------------------------------------------
# Figure output directory
# ------------------------------------------------------------

FIGURE_DIR = (
    ELECTRONIC_STRUCTURE_DIR
    / "figures"
)

# ------------------------------------------------------------
# VASP vasprun.xml paths
# ------------------------------------------------------------

VASPRUN_PATHS = {
    structure_name: (
        RESULTS_ROOT
        / VACANCY_FOLDER_MAP[structure_name]
        / "scf"
        / "vasprun.xml"
    )
    for structure_name in VACANCY_STRUCTURES
}

# ============================================================
# CSV helpers
# ============================================================

def read_csv_rows(
    file_path: Path,
) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""

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
    file_path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write dictionaries to a CSV file."""

    if not rows:
        raise ValueError(
            f"No rows are available for {file_path.name}."
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
# PDOS helpers
# ============================================================

def sum_spin_densities(
    densities: dict[Spin, np.ndarray],
) -> np.ndarray:
    """
    Sum all available spin channels.

    Your current calculations use ISPIN = 1, but this also works
    if spin-polarized data are analyzed later.
    """

    if not densities:
        raise ValueError(
            "The DOS object contains no density data."
        )

    total_density = np.zeros_like(
        next(iter(densities.values())),
        dtype=float,
    )

    for spin_density in densities.values():
        total_density += np.asarray(
            spin_density,
            dtype=float,
        )

    return total_density


def group_targets_by_structure(
    target_rows: list[dict[str, str]],
) -> dict[
    str,
    dict[str, list[dict[str, str]]],
]:
    """
    Group target rows as:

        structure
            coordination type
                target rows
    """

    grouped: dict[
        str,
        dict[str, list[dict[str, str]]],
    ] = defaultdict(
        lambda: defaultdict(list)
    )

    required_columns = {
        "structure",
        "coordination_type",
        "pdos_python_index",
        "defect_ga_atom_number",
    }

    missing_columns = (
        required_columns
        - set(target_rows[0].keys())
    )

    if missing_columns:
        raise KeyError(
            "pdos_targets.csv is missing required columns:\n"
            + ", ".join(sorted(missing_columns))
        )

    for row in target_rows:
        structure_name = row["structure"]
        coordination_type = row[
            "coordination_type"
        ]

        if structure_name not in VACANCY_STRUCTURES:
            continue

        if coordination_type not in COORDINATION_TYPES:
            continue

        grouped[
            structure_name
        ][coordination_type].append(row)

    return grouped


def validate_site_index(
    structure_name: str,
    structure,
    python_index: int,
) -> None:
    """Check that a target index exists and corresponds to Ga."""

    if not 0 <= python_index < len(structure):
        raise IndexError(
            f"{structure_name}: Python index {python_index} "
            f"is outside a structure containing "
            f"{len(structure)} atoms."
        )

    site = structure[python_index]

    if site.species_string != "Ga":
        raise ValueError(
            f"{structure_name}: Python index {python_index} "
            f"is {site.species_string}, not Ga."
        )


def extract_average_group_pdos(
    complete_dos,
    target_rows: list[dict[str, str]],
) -> dict[OrbitalType, np.ndarray]:
    """
    Average site-projected s, p, and d DOS over selected Ga atoms.
    """

    structure = complete_dos.structure

    orbital_sums: dict[
        OrbitalType,
        np.ndarray,
    ] = {}

    for target_row in target_rows:
        python_index = int(
            target_row["pdos_python_index"]
        )

        validate_site_index(
            structure_name=
                target_row["structure"],
            structure=structure,
            python_index=python_index,
        )

        site = structure[python_index]

        site_spd_dos = (
            complete_dos.get_site_spd_dos(
                site
            )
        )

        for orbital_type in ORBITALS:
            if orbital_type in site_spd_dos:
                density = sum_spin_densities(
                    site_spd_dos[
                        orbital_type
                    ].densities
                )
            else:
                density = np.zeros_like(
                    complete_dos.energies,
                    dtype=float,
                )

            if orbital_type not in orbital_sums:
                orbital_sums[
                    orbital_type
                ] = np.zeros_like(
                    density,
                    dtype=float,
                )

            orbital_sums[
                orbital_type
            ] += density

    number_of_sites = len(target_rows)

    if number_of_sites == 0:
        raise ValueError(
            "Cannot average PDOS over zero target sites."
        )

    if NORMALIZE_BY_NUMBER_OF_SITES:
        for orbital_type in orbital_sums:
            orbital_sums[
                orbital_type
            ] /= number_of_sites

    return orbital_sums


# ============================================================
# Analyze one vacancy
# ============================================================

def analyze_vacancy_pdos(
    structure_name: str,
    coordination_groups: dict[
        str,
        list[dict[str, str]],
    ],
) -> dict[str, object]:
    """Extract average tetrahedral and octahedral Ga PDOS."""

    vasprun_path = VASPRUN_PATHS[
        structure_name
    ]

    if not vasprun_path.is_file():
        raise FileNotFoundError(
            f"Cannot find vasprun.xml:\n"
            f"{vasprun_path}"
        )

    vasprun = Vasprun(
        vasprun_path,
        parse_dos=True,
        parse_eigen=False,
        parse_projected_eigen=False,
    )

    complete_dos = vasprun.complete_dos

    if complete_dos is None:
        raise ValueError(
            f"{structure_name}: complete DOS is unavailable."
        )

    if not complete_dos.pdos:
        raise ValueError(
            f"{structure_name}: projected DOS is unavailable."
        )

    fermi_energy = float(
        complete_dos.efermi
    )

    absolute_energies = np.asarray(
        complete_dos.energies,
        dtype=float,
    )

    relative_energies = (
        absolute_energies
        - fermi_energy
    )

    group_pdos: dict[
        str,
        dict[OrbitalType, np.ndarray],
    ] = {}

    group_atom_numbers: dict[
        str,
        list[int],
    ] = {}

    for coordination_type in COORDINATION_TYPES:
        target_rows = coordination_groups.get(
            coordination_type,
            [],
        )

        if not target_rows:
            continue

        group_pdos[
            coordination_type
        ] = extract_average_group_pdos(
            complete_dos=complete_dos,
            target_rows=target_rows,
        )

        group_atom_numbers[
            coordination_type
        ] = [
            int(
                row[
                    "defect_ga_atom_number"
                ]
            )
            for row in target_rows
        ]

    if not group_pdos:
        raise ValueError(
            f"{structure_name}: no valid Ga PDOS targets "
            "were found."
        )

    return {
        "structure": structure_name,
        "label": VACANCY_LABELS[
            structure_name
        ],
        "fermi_energy_eV": fermi_energy,
        "energies_absolute_eV":
            absolute_energies,
        "energies_relative_eV":
            relative_energies,
        "group_pdos": group_pdos,
        "group_atom_numbers":
            group_atom_numbers,
        "electronic_converged":
            bool(
                vasprun.converged_electronic
            ),
        "overall_converged":
            bool(
                vasprun.converged
            ),
    }


# ============================================================
# CSV output
# ============================================================

def build_pdos_csv_rows(
    result: dict[str, object],
) -> list[dict[str, object]]:
    """Build long-format PDOS rows for one vacancy."""

    rows: list[dict[str, object]] = []

    absolute_energies = np.asarray(
        result["energies_absolute_eV"],
        dtype=float,
    )

    relative_energies = np.asarray(
        result["energies_relative_eV"],
        dtype=float,
    )

    group_pdos = result["group_pdos"]
    group_atom_numbers = result[
        "group_atom_numbers"
    ]

    for coordination_type, orbital_data in (
        group_pdos.items()
    ):
        atom_numbers = group_atom_numbers[
            coordination_type
        ]

        atom_number_text = ";".join(
            str(number)
            for number in atom_numbers
        )

        for (
            absolute_energy,
            relative_energy,
            point_index,
        ) in zip(
            absolute_energies,
            relative_energies,
            range(len(relative_energies)),
        ):
            row: dict[str, object] = {
                "structure":
                    result["structure"],
                "vacancy_label":
                    result["label"],
                "coordination_type":
                    coordination_type,
                "number_of_ga_atoms":
                    len(atom_numbers),
                "ga_atom_numbers":
                    atom_number_text,
                "fermi_energy_eV":
                    result["fermi_energy_eV"],
                "energy_absolute_eV":
                    float(absolute_energy),
                "energy_minus_fermi_eV":
                    float(relative_energy),
            }

            for orbital_type in ORBITALS:
                row[
                    f"ga_{ORBITAL_LABELS[orbital_type]}_pdos"
                ] = float(
                    orbital_data[
                        orbital_type
                    ][point_index]
                )

            rows.append(row)

    return rows


# ============================================================
# Plotting
# ============================================================

def plot_vacancy_pdos(
    result: dict[str, object],
    output_path: Path,
) -> None:
    """
    Plot average tetrahedral and octahedral Ga s/p/d PDOS.

    Solid lines:
        tetrahedral Ga

    Dashed lines:
        octahedral Ga
    """

    energies = np.asarray(
        result["energies_relative_eV"],
        dtype=float,
    )

    group_pdos = result["group_pdos"]
    group_atom_numbers = result[
        "group_atom_numbers"
    ]

    visible_mask = (
        (energies >= ENERGY_MIN_EV)
        & (energies <= ENERGY_MAX_EV)
    )

    if not np.any(visible_mask):
        raise ValueError(
            f"No DOS data in the selected energy range "
            f"{ENERGY_MIN_EV} to {ENERGY_MAX_EV} eV."
        )

    figure, axis = plt.subplots(
        figsize=(9, 5.8),
    )

    line_styles = {
        "tetrahedral": "-",
        "octahedral": "--",
    }

    maximum_visible_density = 0.0

    for coordination_type in COORDINATION_TYPES:
        if coordination_type not in group_pdos:
            continue

        atom_numbers = group_atom_numbers[
            coordination_type
        ]

        atom_text = ", ".join(
            f"Ga{number}"
            for number in atom_numbers
        )

        for orbital_type in ORBITALS:
            density = np.asarray(
                group_pdos[
                    coordination_type
                ][orbital_type],
                dtype=float,
            )

            orbital_label = ORBITAL_LABELS[
                orbital_type
            ]

            axis.plot(
                energies[visible_mask],
                density[visible_mask],
                linestyle=line_styles[
                    coordination_type
                ],
                linewidth=1.7,
                label=(
                    f"{coordination_type.capitalize()} "
                    f"Ga-{orbital_label} "
                    f"({atom_text})"
                ),
            )

            maximum_visible_density = max(
                maximum_visible_density,
                float(
                    np.max(
                        density[
                            visible_mask
                        ]
                    )
                ),
            )

    axis.axvline(
        0.0,
        linestyle=":",
        linewidth=1.1,
        label=r"$E_F$",
    )

    axis.set_xlim(
        ENERGY_MIN_EV,
        ENERGY_MAX_EV,
    )

    axis.set_ylim(
        0,
        (
            maximum_visible_density * 1.10
            if maximum_visible_density > 0
            else 1.0
        ),
    )

    axis.set_xlabel(
        r"Energy, $E-E_F$ (eV)"
    )

    axis.set_ylabel(
        "Average Ga PDOS (states/eV/site)"
    )

    axis.set_title(
        f"First-shell Ga PDOS: "
        f"{result['label']}"
    )

    axis.grid(
        alpha=0.2,
    )

    axis.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Terminal summary
# ============================================================

def print_result_summary(
    result: dict[str, object],
) -> None:
    """Print information about one vacancy PDOS result."""

    print()
    print(result["structure"])
    print("-" * 78)

    print(
        "  Fermi energy         : "
        f"{float(result['fermi_energy_eV']):.6f} eV"
    )

    print(
        "  Electronic converged : "
        f"{result['electronic_converged']}"
    )

    print(
        "  Overall converged    : "
        f"{result['overall_converged']}"
    )

    for coordination_type, atom_numbers in (
        result["group_atom_numbers"].items()
    ):
        atom_text = ", ".join(
            f"Ga{number}"
            for number in atom_numbers
        )

        print(
            f"  {coordination_type.capitalize():12s}: "
            f"{atom_text}"
        )


# ============================================================
# Main program
# ============================================================

def main() -> None:
    """Extract and plot first-shell Ga PDOS."""

    target_rows = read_csv_rows(
        TARGETS_CSV_PATH
    )

    grouped_targets = (
        group_targets_by_structure(
            target_rows
        )
    )

    print("=" * 78)
    print("First-shell Ga projected DOS")
    print("=" * 78)

    generated_files: list[Path] = []

    for structure_name in VACANCY_STRUCTURES:
        if structure_name not in grouped_targets:
            print()
            print(
                f"{structure_name}: no PDOS targets found; skipping."
            )
            continue

        result = analyze_vacancy_pdos(
            structure_name=structure_name,
            coordination_groups=
                grouped_targets[
                    structure_name
                ],
        )

        csv_rows = build_pdos_csv_rows(
            result
        )

        csv_output_path = (
            PDOS_OUTPUT_DIR
            / f"{structure_name}_ga_pdos.csv"
        )

        figure_output_path = (
            FIGURE_DIR
            / f"{structure_name}_ga_pdos.png"
        )

        write_csv(
            file_path=csv_output_path,
            rows=csv_rows,
        )

        plot_vacancy_pdos(
            result=result,
            output_path=
                figure_output_path,
        )

        generated_files.extend(
            [
                csv_output_path,
                figure_output_path,
            ]
        )

        print_result_summary(
            result
        )

    print()
    print("=" * 78)
    print("Output files")
    print("=" * 78)

    for file_path in generated_files:
        print(file_path)

    print()
    print(
        "Solid lines represent tetrahedral Ga; dashed lines represent "
        "octahedral Ga. Multiple Ga sites of the same coordination "
        "type are averaged before plotting."
    )


if __name__ == "__main__":
    main()