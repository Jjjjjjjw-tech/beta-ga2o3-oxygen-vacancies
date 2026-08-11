#!/usr/bin/env python3
"""
Analyze and compare the total density of states (DOS) of pristine
and oxygen-vacancy beta-Ga2O3 structures.

Inputs
------
beta-Ga2O3_oxygen_vacancy_VASP_inputs/
    received_results/
        2026-07-25_results/
            pristine/scf/vasprun.xml
            vacancy_O1/scf/vasprun.xml
            vacancy_O2/scf/vasprun.xml
            vacancy_O3/scf/vasprun.xml

Outputs
-------
analysis/electronic_structure/
    total_dos.csv

analysis/electronic_structure/figures/
    total_dos_comparison.png
    total_dos_near_fermi.png

Notes
-----
1. Each energy axis is shifted independently so that:

       E - E_F = 0 eV

2. DOS values are divided by the number of atoms in each structure.
   The plotted unit is therefore states / eV / atom.

3. The current calculations use ISPIN = 1, so only one total DOS
   curve is expected for each structure.

4. Fermi-level alignment is useful for an initial comparison, but
   the Fermi level of a semiconductor can be somewhat arbitrary
   inside the band gap. A later analysis may align structures using
   the valence-band maximum or an electrostatic reference.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Vasprun


# ============================================================
# Configuration
# ============================================================

RESULT_DATE = "2026-07-25_results"

STRUCTURE_NAMES = (
    "pristine",
    "vacancy_O1",
    "vacancy_O2",
    "vacancy_O3",
)

STRUCTURE_LABELS = {
    "pristine": "Pristine",
    "vacancy_O1": "O1 vacancy",
    "vacancy_O2": "O2 vacancy",
    "vacancy_O3": "O3 vacancy",
}

# Energy window for the full comparison figure.
FULL_ENERGY_MIN_EV = -10.0
FULL_ENERGY_MAX_EV = 8.0

# Energy window near the Fermi level.
NEAR_FERMI_MIN_EV = -3.0
NEAR_FERMI_MAX_EV = 3.0

# Divide total DOS by number of atoms.
NORMALIZE_PER_ATOM = True


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

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "analysis"
    / "electronic_structure"
)

FIGURE_DIR = (
    OUTPUT_ROOT
    / "figures"
)

DOS_CSV_OUTPUT_PATH = (
    OUTPUT_ROOT
    / "total_dos.csv"
)

FULL_DOS_FIGURE_PATH = (
    FIGURE_DIR
    / "total_dos_comparison.png"
)

NEAR_FERMI_FIGURE_PATH = (
    FIGURE_DIR
    / "total_dos_near_fermi.png"
)


VASPRUN_PATHS = {
    structure_name: (
        RESULTS_ROOT
        / structure_name
        / "scf"
        / "vasprun.xml"
    )
    for structure_name in STRUCTURE_NAMES
}


# ============================================================
# General helper functions
# ============================================================

def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write dictionaries to a CSV file."""

    if not rows:
        raise ValueError(
            f"No data rows are available for {output_path.name}."
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


def sum_spin_densities(
    densities: dict[Spin, np.ndarray],
) -> np.ndarray:
    """
    Sum all available spin-channel densities.

    For ISPIN = 1, pymatgen normally stores only Spin.up.
    This function also works if spin-polarized calculations
    are analyzed later.
    """

    if not densities:
        raise ValueError(
            "The DOS object does not contain density data."
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


# ============================================================
# Read DOS data
# ============================================================

def read_total_dos(
    structure_name: str,
    vasprun_path: Path,
) -> dict[str, object]:
    """
    Read total DOS and convergence information from vasprun.xml.
    """

    if not vasprun_path.is_file():
        raise FileNotFoundError(
            f"{structure_name}: cannot find vasprun.xml:\n"
            f"{vasprun_path}"
        )

    vasprun = Vasprun(
        vasprun_path,
        parse_dos=True,
        parse_eigen=False,
        parse_projected_eigen=False,
    )

    # Vasprun.tdos is the total DOS object.
    total_dos = vasprun.tdos

    if total_dos is None:
        raise ValueError(
            f"{structure_name}: no total DOS was found "
            f"in vasprun.xml:\n{vasprun_path}"
        )

    energies_absolute = np.asarray(
        total_dos.energies,
        dtype=float,
    )

    fermi_energy = float(
        total_dos.efermi
    )

    energies_relative = (
        energies_absolute
        - fermi_energy
    )

    densities = sum_spin_densities(
        total_dos.densities
    )

    final_structure = vasprun.final_structure

    if final_structure is None:
        raise ValueError(
            f"{structure_name}: final structure was not found "
            f"in vasprun.xml."
        )

    number_of_atoms = len(
        final_structure
    )

    if NORMALIZE_PER_ATOM:
        densities = (
            densities
            / number_of_atoms
        )

    return {
        "structure": structure_name,
        "label": STRUCTURE_LABELS[
            structure_name
        ],
        "number_of_atoms": number_of_atoms,
        "fermi_energy_eV": fermi_energy,
        "electronic_converged":
            bool(
                vasprun.converged_electronic
            ),
        "overall_converged":
            bool(
                vasprun.converged
            ),
        "energies_absolute_eV":
            energies_absolute,
        "energies_relative_eV":
            energies_relative,
        "dos":
            densities,
    }

def read_all_dos() -> list[dict[str, object]]:
    """Read total DOS for all configured structures."""

    results: list[
        dict[str, object]
    ] = []

    print("=" * 78)
    print("Total DOS data")
    print("=" * 78)

    for structure_name in STRUCTURE_NAMES:
        dos_result = read_total_dos(
            structure_name=structure_name,
            vasprun_path=
                VASPRUN_PATHS[
                    structure_name
                ],
        )

        results.append(
            dos_result
        )

        print()
        print(structure_name)
        print(
            "  Number of atoms       : "
            f"{dos_result['number_of_atoms']}"
        )
        print(
            "  Fermi energy          : "
            f"{dos_result['fermi_energy_eV']:.6f} eV"
        )
        print(
            "  Electronic converged  : "
            f"{dos_result['electronic_converged']}"
        )
        print(
            "  Overall converged     : "
            f"{dos_result['overall_converged']}"
        )
        print(
            "  DOS energy points     : "
            f"{len(dos_result['energies_relative_eV'])}"
        )

    return results


# ============================================================
# Prepare CSV output
# ============================================================

def build_dos_csv_rows(
    dos_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    """
    Build long-format DOS rows.

    Each row corresponds to one energy point of one structure.
    """

    rows: list[
        dict[str, object]
    ] = []

    for result in dos_results:
        structure_name = str(
            result["structure"]
        )

        label = str(
            result["label"]
        )

        number_of_atoms = int(
            result["number_of_atoms"]
        )

        fermi_energy = float(
            result["fermi_energy_eV"]
        )

        absolute_energies = np.asarray(
            result[
                "energies_absolute_eV"
            ],
            dtype=float,
        )

        relative_energies = np.asarray(
            result[
                "energies_relative_eV"
            ],
            dtype=float,
        )

        densities = np.asarray(
            result["dos"],
            dtype=float,
        )

        for (
            absolute_energy,
            relative_energy,
            density,
        ) in zip(
            absolute_energies,
            relative_energies,
            densities,
        ):
            rows.append(
                {
                    "structure":
                        structure_name,
                    "label":
                        label,
                    "number_of_atoms":
                        number_of_atoms,
                    "fermi_energy_eV":
                        fermi_energy,
                    "energy_absolute_eV":
                        float(
                            absolute_energy
                        ),
                    "energy_minus_fermi_eV":
                        float(
                            relative_energy
                        ),
                    "dos_states_per_eV_per_atom":
                        float(
                            density
                        ),
                }
            )

    return rows


# ============================================================
# Plotting
# ============================================================

def plot_total_dos(
    dos_results: list[dict[str, object]],
    output_path: Path,
    energy_min_eV: float,
    energy_max_eV: float,
    title: str,
) -> None:
    """Plot total DOS curves within a selected energy range."""

    figure, axis = plt.subplots(
        figsize=(8, 5.5),
    )

    maximum_visible_dos = 0.0

    for result in dos_results:
        energies = np.asarray(
            result[
                "energies_relative_eV"
            ],
            dtype=float,
        )

        densities = np.asarray(
            result["dos"],
            dtype=float,
        )

        visible_mask = (
            (energies >= energy_min_eV)
            & (energies <= energy_max_eV)
        )

        if not np.any(
            visible_mask
        ):
            raise ValueError(
                f"No DOS points for "
                f"{result['structure']} in the range "
                f"{energy_min_eV} to {energy_max_eV} eV."
            )

        axis.plot(
            energies[
                visible_mask
            ],
            densities[
                visible_mask
            ],
            linewidth=1.7,
            label=str(
                result["label"]
            ),
        )

        visible_maximum = float(
            np.max(
                densities[
                    visible_mask
                ]
            )
        )

        maximum_visible_dos = max(
            maximum_visible_dos,
            visible_maximum,
        )

    axis.axvline(
        0.0,
        linewidth=1.0,
        linestyle="--",
        label=r"$E_F$",
    )

    axis.set_xlim(
        energy_min_eV,
        energy_max_eV,
    )

    axis.set_ylim(
        0,
        maximum_visible_dos * 1.08
        if maximum_visible_dos > 0
        else 1,
    )

    axis.set_xlabel(
        r"Energy, $E-E_F$ (eV)"
    )

    if NORMALIZE_PER_ATOM:
        axis.set_ylabel(
            "DOS (states/eV/atom)"
        )
    else:
        axis.set_ylabel(
            "DOS (states/eV)"
        )

    axis.set_title(
        title
    )

    axis.legend(
        frameon=False,
    )

    axis.grid(
        alpha=0.2,
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

    plt.close(
        figure
    )


# ============================================================
# Main program
# ============================================================

def main() -> None:
    """Read, export, and plot the total DOS."""

    dos_results = read_all_dos()

    dos_rows = build_dos_csv_rows(
        dos_results
    )

    write_csv(
        output_path=
            DOS_CSV_OUTPUT_PATH,
        rows=dos_rows,
    )

    plot_total_dos(
        dos_results=dos_results,
        output_path=
            FULL_DOS_FIGURE_PATH,
        energy_min_eV=
            FULL_ENERGY_MIN_EV,
        energy_max_eV=
            FULL_ENERGY_MAX_EV,
        title=(
            "Total DOS of pristine and "
            "oxygen-vacancy structures"
        ),
    )

    plot_total_dos(
        dos_results=dos_results,
        output_path=
            NEAR_FERMI_FIGURE_PATH,
        energy_min_eV=
            NEAR_FERMI_MIN_EV,
        energy_max_eV=
            NEAR_FERMI_MAX_EV,
        title=(
            "Total DOS near the Fermi level"
        ),
    )

    print()
    print("=" * 78)
    print("Output files")
    print("=" * 78)

    print(
        DOS_CSV_OUTPUT_PATH
    )
    print(
        FULL_DOS_FIGURE_PATH
    )
    print(
        NEAR_FERMI_FIGURE_PATH
    )

    print()
    print(
        "Note: the curves are independently aligned to E_F = 0 eV. "
        "For semiconductors, the precise Fermi-level position inside "
        "the gap can depend on the VASP smearing and DOS settings. "
        "Interpret this first comparison qualitatively."
    )


if __name__ == "__main__":
    main()