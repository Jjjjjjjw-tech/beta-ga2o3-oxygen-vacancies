#!/usr/bin/env python3

"""
Analyze and compare the total density of states (DOS) of pristine
and oxygen-vacancy beta-Ga2O3 structures.

This version uses the newly imported external VASP dataset defined
in the project-level config.py.

Systems
-------
- pristine
- vacancy_O1
- vacancy_O2
- vacancy_O3

Energy alignment
----------------
For each system, the energy axis is shifted independently so that

    E - E_F = 0 eV

where E_F is the Fermi energy of that calculation.

DOS normalization
-----------------
The total DOS is divided by the number of atoms in each structure,
giving units of

    states / eV / atom

This allows the 80-atom pristine structure and 79-atom vacancy
structures to be compared directly.

Outputs
-------
analysis/electronic_structure/
    total_dos.csv

analysis/electronic_structure/figures/
    total_dos_comparison.png
    total_dos_near_fermi.png
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Vasprun


# ============================================================
# Import project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from config import (
    PRISTINE_SCF_DIR,
    VACANCY_O1_SCF_DIR,
    VACANCY_O2_SCF_DIR,
    VACANCY_O3_SCF_DIR,
    ELECTRONIC_STRUCTURE_DIR,
    ELECTRONIC_STRUCTURE_FIGURE_DIR,
)


# ============================================================
# Calculation directories
# ============================================================

CALCULATIONS = {
    "pristine": PRISTINE_SCF_DIR,
    "vacancy_O1": VACANCY_O1_SCF_DIR,
    "vacancy_O2": VACANCY_O2_SCF_DIR,
    "vacancy_O3": VACANCY_O3_SCF_DIR,
}


# ============================================================
# Plot labels
# ============================================================

PLOT_LABELS = {
    "pristine": "Pristine",
    "vacancy_O1": "O1 vacancy",
    "vacancy_O2": "O2 vacancy",
    "vacancy_O3": "O3 vacancy",
}


# ============================================================
# Output files
# ============================================================

OUTPUT_CSV = (
    ELECTRONIC_STRUCTURE_DIR
    / "total_dos.csv"
)

OUTPUT_COMPARISON_FIGURE = (
    ELECTRONIC_STRUCTURE_FIGURE_DIR
    / "total_dos_comparison.png"
)

OUTPUT_NEAR_FERMI_FIGURE = (
    ELECTRONIC_STRUCTURE_FIGURE_DIR
    / "total_dos_near_fermi.png"
)


# ============================================================
# Ensure output directories exist
# ============================================================

ELECTRONIC_STRUCTURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ELECTRONIC_STRUCTURE_FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Helper functions
# ============================================================

def print_header(title: str) -> None:
    """Print a formatted terminal header."""

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def get_total_density(complete_dos) -> np.ndarray:
    """
    Return total DOS summed over all spin channels.

    Works for both ISPIN = 1 and ISPIN = 2 calculations.
    """

    total_density = np.zeros(
        len(complete_dos.energies),
        dtype=float,
    )

    for density in complete_dos.densities.values():
        total_density += np.asarray(
            density,
            dtype=float,
        )

    return total_density


def read_dos(
    name: str,
    scf_dir: Path,
) -> dict:
    """
    Read total DOS information from vasprun.xml.

    Parameters
    ----------
    name
        Calculation label.

    scf_dir
        SCF calculation directory.

    Returns
    -------
    dict
        Electronic-structure information.
    """

    vasprun_path = (
        scf_dir
        / "vasprun.xml"
    )

    if not vasprun_path.exists():
        raise FileNotFoundError(
            f"{name}: vasprun.xml not found:\n"
            f"{vasprun_path}"
        )

    print(
        f"Reading {name} ..."
    )

    vasprun = Vasprun(
        vasprun_path,
        parse_dos=True,
        parse_eigen=False,
        parse_projected_eigen=False,
    )

    complete_dos = (
        vasprun.complete_dos
    )

    number_of_atoms = len(
        vasprun.final_structure
    )

    efermi = float(
        vasprun.efermi
    )

    # Shift energy axis so EF = 0 eV
    energies = (
        np.asarray(
            complete_dos.energies,
            dtype=float,
        )
        - efermi
    )

    total_density = (
        get_total_density(
            complete_dos
        )
    )

    # Normalize by number of atoms
    dos_per_atom = (
        total_density
        / number_of_atoms
    )

    band_gap = float(
        complete_dos.get_gap()
    )

    return {
        "name": name,
        "label": PLOT_LABELS[name],
        "number_of_atoms": number_of_atoms,
        "efermi": efermi,
        "band_gap": band_gap,
        "energies": energies,
        "dos_per_atom": dos_per_atom,
        "electronic_converged":
            vasprun.converged_electronic,
        "overall_converged":
            vasprun.converged,
    }


# ============================================================
# Read all calculations
# ============================================================

def read_all_dos() -> dict:
    """Read DOS data for all four structures."""

    results = {}

    for name, scf_dir in CALCULATIONS.items():

        results[name] = read_dos(
            name,
            scf_dir,
        )

    return results


# ============================================================
# Save CSV
# ============================================================

def save_dos_csv(
    results: dict,
) -> None:
    """
    Save total DOS data in long-table format.

    Each row contains one energy point for one structure.
    """

    rows = []

    for name, result in results.items():

        for energy, density in zip(
            result["energies"],
            result["dos_per_atom"],
        ):

            rows.append(
                {
                    "structure": name,
                    "label": result["label"],
                    "energy_minus_fermi_eV":
                        energy,
                    "dos_states_per_eV_per_atom":
                        density,
                    "fermi_energy_eV":
                        result["efermi"],
                    "dos_band_gap_eV":
                        result["band_gap"],
                    "number_of_atoms":
                        result["number_of_atoms"],
                }
            )

    dataframe = pd.DataFrame(
        rows
    )

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
    )

# ============================================================
# Plot: full DOS comparison
# ============================================================

def plot_total_dos(
    results: dict,
) -> None:
    """Plot total DOS over the full energy range."""

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for result in results.values():

        ax.plot(
            result["energies"],
            result["dos_per_atom"],
            linewidth=1.8,
            label=result["label"],
        )

    # Fermi level
    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        label=r"$E_F$",
    )

    ax.set_xlim(
        -10,
        8,
    )

    # Automatic y-axis scaling
    max_dos = max(
        np.max(result["dos_per_atom"])
        for result in results.values()
    )

    ax.set_ylim(
        0,
        1.15,
    )


    ax.set_xlabel(
        r"Energy, $E-E_F$ (eV)"
    )

    ax.set_ylabel(
        "DOS (states/eV/atom)"
    )

    ax.set_title(
        "Total DOS of pristine and oxygen-vacancy structures"
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_COMPARISON_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Plot: near-Fermi region
# ============================================================

def plot_near_fermi(
    results: dict,
) -> None:
    """Plot DOS around the Fermi level."""

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    max_dos = 0.0

    for result in results.values():

        energies = result["energies"]
        dos = result["dos_per_atom"]

        mask = (
            (energies >= -3.0)
            & (energies <= 3.0)
        )

        ax.plot(
            energies[mask],
            dos[mask],
            linewidth=1.8,
            label=result["label"],
        )

        if np.any(mask):
            max_dos = max(
                max_dos,
                np.max(dos[mask]),
            )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        label=r"$E_F$",
    )

    ax.set_xlim(
        -3,
        3,
    )

    ax.set_ylim(
        0,
        max_dos * 1.10,
    )

    ax.set_xlabel(
        r"Energy, $E-E_F$ (eV)"
    )

    ax.set_ylabel(
        "DOS (states/eV/atom)"
    )

    ax.set_title(
        "Total DOS near the Fermi level"
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_NEAR_FERMI_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

# ============================================================
# Print summary
# ============================================================

def print_summary(
    results: dict,
) -> None:
    """Print calculation and DOS summary."""

    print_header(
        "Total DOS summary"
    )

    for name, result in results.items():

        print(
            f"{name}"
        )

        print(
            f"  Number of atoms       : "
            f"{result['number_of_atoms']}"
        )

        print(
            f"  Fermi energy          : "
            f"{result['efermi']:.6f} eV"
        )

        print(
            f"  DOS band gap          : "
            f"{result['band_gap']:.6f} eV"
        )

        print(
            f"  Electronic converged  : "
            f"{result['electronic_converged']}"
        )

        print(
            f"  Overall converged     : "
            f"{result['overall_converged']}"
        )

        print()

    print_header(
        "Output files"
    )

    print(
        OUTPUT_CSV
    )

    print(
        OUTPUT_COMPARISON_FIGURE
    )

    print(
        OUTPUT_NEAR_FERMI_FIGURE
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print_header(
        "Total DOS analysis"
    )

    print(
        "Dataset:"
    )

    for name, folder in CALCULATIONS.items():

        print(
            f"  {name:12s}: "
            f"{folder}"
        )

    print()

    results = read_all_dos()

    save_dos_csv(
        results
    )

    plot_total_dos(
        results
    )

    plot_near_fermi(
        results
    )

    print_summary(
        results
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()