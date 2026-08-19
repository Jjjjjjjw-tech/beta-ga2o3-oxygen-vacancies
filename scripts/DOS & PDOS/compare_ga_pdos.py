#!/usr/bin/env python3
"""
Compare matched first-shell Ga projected DOS (PDOS)
between pristine beta-Ga2O3 and oxygen-vacancy structures.

The atom mapping is read from:

analysis/electronic_structure/pdos/pdos_targets.csv

For each first-shell Ga atom, the script compares:

    pristine reference Ga
            vs.
    corresponding Ga in vacancy structure

The s, p, and d orbital contributions are summed separately.

Inputs
------
analysis/electronic_structure/pdos/pdos_targets.csv

beta-Ga2O3_oxygen_vacancy_VASP_inputs/
    received_results/
        2026-08-17_results/
            pristine/scf/vasprun.xml
            1/scf/vasprun.xml
            2/scf/vasprun.xml
            3/scf/vasprun.xml

Outputs
-------
analysis/electronic_structure/pdos/comparison/
    vacancy_O1_ga_pdos_comparison.csv
    vacancy_O2_ga_pdos_comparison.csv
    vacancy_O3_ga_pdos_comparison.csv

analysis/electronic_structure/figures/
    vacancy_O1_ga_pdos_pristine_comparison.png
    vacancy_O2_ga_pdos_pristine_comparison.png
    vacancy_O3_ga_pdos_pristine_comparison.png

Notes
-----
1. Each pristine and vacancy energy axis is shifted independently so that
   its own Fermi level is at E - E_F = 0 eV.

2. The plotted energy window is restricted to -6 to +6 eV.

3. The y-axis is scaled only using PDOS values inside the displayed
   energy window. Strong deep-energy states outside the plotted window
   therefore do not artificially enlarge the y-axis.

4. This script compares PDOS shape, intensity, and orbital character.
   It does not perform absolute energy alignment between pristine and
   defect structures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pymatgen.electronic_structure.core import Orbital
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

# Displayed energy window relative to each structure's own Fermi level
ENERGY_MIN_EV = -6.0
ENERGY_MAX_EV = 6.0

# Small extra margin above the largest visible PDOS peak
Y_AXIS_MARGIN = 1.10


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

PDOS_ROOT = (
    ELECTRONIC_STRUCTURE_DIR
    / "pdos"
)

TARGETS_CSV_PATH = (
    PDOS_ROOT
    / "pdos_targets.csv"
)

COMPARISON_OUTPUT_DIR = (
    PDOS_ROOT
    / "comparison"
)

FIGURE_DIR = (
    ELECTRONIC_STRUCTURE_DIR
    / "figures"
)

PRISTINE_VASPRUN_PATH = (
    RESULTS_ROOT
    / "pristine"
    / "scf"
    / "vasprun.xml"
)

VACANCY_VASPRUN_PATHS = {
    structure_name: (
        RESULTS_ROOT
        / VACANCY_FOLDER_MAP[structure_name]
        / "scf"
        / "vasprun.xml"
    )
    for structure_name in VACANCY_STRUCTURES
}


# ============================================================
# Orbital groups
# ============================================================

S_ORBITALS = (
    Orbital.s,
)

P_ORBITALS = (
    Orbital.py,
    Orbital.pz,
    Orbital.px,
)

D_ORBITALS = (
    Orbital.dxy,
    Orbital.dyz,
    Orbital.dz2,
    Orbital.dxz,
    Orbital.dx2,
)

ORBITAL_GROUPS = {
    "s": S_ORBITALS,
    "p": P_ORBITALS,
    "d": D_ORBITALS,
}


# ============================================================
# Helper functions
# ============================================================

def read_vasprun(path: Path) -> Vasprun:
    """
    Read a VASP vasprun.xml file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"vasprun.xml not found:\n{path}"
        )

    return Vasprun(
        path,
        parse_projected_eigen=True,
        parse_potcar_file=False,
    )


def atom_number_to_python_index(
    atom_number: int,
) -> int:
    """
    Convert one-based atom numbering to zero-based Python indexing.
    """

    return int(atom_number) - 1


def get_orbital_group_pdos(
    complete_dos,
    site_index: int,
    orbitals,
) -> np.ndarray:
    """
    Sum the PDOS of selected orbitals for one atomic site.

    Parameters
    ----------
    complete_dos
        Pymatgen CompleteDos object.

    site_index
        Zero-based Python index of the atomic site.

    orbitals
        Tuple of pymatgen Orbital objects.

    Returns
    -------
    np.ndarray
        Summed PDOS over all selected orbitals and spin channels.
    """

    site = complete_dos.structure[site_index]

    site_pdos = complete_dos.pdos[site]

    total = np.zeros_like(
        complete_dos.energies,
        dtype=float,
    )

    for orbital in orbitals:

        if orbital not in site_pdos:
            continue

        orbital_data = site_pdos[orbital]

        for values in orbital_data.values():
            total += np.asarray(
                values,
                dtype=float,
            )

    return total


def get_visible_maximum(
    energies: np.ndarray,
    pdos: np.ndarray,
) -> float:
    """
    Return the maximum PDOS inside the displayed energy window.

    This avoids deep-energy peaks outside the plotted range from
    artificially enlarging the y-axis.
    """

    mask = (
        (energies >= ENERGY_MIN_EV)
        & (energies <= ENERGY_MAX_EV)
    )

    if not np.any(mask):
        return 0.0

    return float(
        np.max(
            pdos[mask]
        )
    )


# ============================================================
# Main analysis
# ============================================================

def main() -> None:

    print("=" * 72)
    print("Matched pristine-vacancy Ga PDOS comparison")
    print("=" * 72)

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    COMPARISON_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Read atom mapping
    # --------------------------------------------------------

    if not TARGETS_CSV_PATH.exists():
        raise FileNotFoundError(
            "pdos_targets.csv not found:\n"
            f"{TARGETS_CSV_PATH}"
        )

    targets = pd.read_csv(
        TARGETS_CSV_PATH
    )

    required_columns = {
        "structure",
        "reference_ga_atom_number",
        "defect_ga_atom_number",
        "coordination_type",
    }

    missing_columns = (
        required_columns
        - set(targets.columns)
    )

    if missing_columns:
        raise ValueError(
            "pdos_targets.csv is missing required columns:\n"
            + "\n".join(
                sorted(
                    missing_columns
                )
            )
        )

    # --------------------------------------------------------
    # Read pristine SCF result
    # --------------------------------------------------------

    print()
    print("Reading pristine vasprun.xml ...")

    pristine_vr = read_vasprun(
        PRISTINE_VASPRUN_PATH
    )

    pristine_dos = (
        pristine_vr.complete_dos
    )

    pristine_energies = (
        pristine_dos.energies
        - pristine_dos.efermi
    )

    print(
        f"Pristine Fermi energy: "
        f"{pristine_dos.efermi:.6f} eV"
    )

    # --------------------------------------------------------
    # Analyze each vacancy structure
    # --------------------------------------------------------

    for structure_name in VACANCY_STRUCTURES:

        print()
        print("=" * 72)
        print(
            VACANCY_LABELS[
                structure_name
            ]
        )
        print("=" * 72)

        rows = targets[
            targets["structure"]
            == structure_name
        ].copy()

        if rows.empty:

            print(
                f"No target Ga atoms found for "
                f"{structure_name}. Skipping."
            )

            continue

        # ----------------------------------------------------
        # Read vacancy SCF result
        # ----------------------------------------------------

        vacancy_vr = read_vasprun(
            VACANCY_VASPRUN_PATHS[
                structure_name
            ]
        )

        vacancy_dos = (
            vacancy_vr.complete_dos
        )

        vacancy_energies = (
            vacancy_dos.energies
            - vacancy_dos.efermi
        )

        print(
            f"Vacancy Fermi energy: "
            f"{vacancy_dos.efermi:.6f} eV"
        )

        # ----------------------------------------------------
        # Create figure
        # ----------------------------------------------------

        number_of_sites = len(rows)

        fig, axes = plt.subplots(
            number_of_sites,
            1,
            figsize=(
                9,
                3.6 * number_of_sites,
            ),
            sharex=True,
        )

        if number_of_sites == 1:
            axes = [axes]

        output_rows = []

        # ----------------------------------------------------
        # Loop over matched Ga atoms
        # ----------------------------------------------------

        for ax, (_, row) in zip(
            axes,
            rows.iterrows(),
        ):

            reference_atom_number = int(
                row[
                    "reference_ga_atom_number"
                ]
            )

            defect_atom_number = int(
                row[
                    "defect_ga_atom_number"
                ]
            )

            coordination_type = str(
                row[
                    "coordination_type"
                ]
            )

            reference_index = (
                atom_number_to_python_index(
                    reference_atom_number
                )
            )

            defect_index = (
                atom_number_to_python_index(
                    defect_atom_number
                )
            )

            print(
                f"Ga{reference_atom_number} "
                f"(pristine) -> "
                f"Ga{defect_atom_number} "
                f"(defect), "
                f"{coordination_type}"
            )

            visible_max = 0.0

            # ------------------------------------------------
            # s / p / d PDOS
            # ------------------------------------------------

            for (
                orbital_name,
                orbitals,
            ) in ORBITAL_GROUPS.items():

                pristine_pdos = (
                    get_orbital_group_pdos(
                        pristine_dos,
                        reference_index,
                        orbitals,
                    )
                )

                vacancy_pdos = (
                    get_orbital_group_pdos(
                        vacancy_dos,
                        defect_index,
                        orbitals,
                    )
                )

                # --------------------------------------------
                # Determine y-axis scale only from visible range
                # --------------------------------------------

                pristine_visible_max = (
                    get_visible_maximum(
                        pristine_energies,
                        pristine_pdos,
                    )
                )

                vacancy_visible_max = (
                    get_visible_maximum(
                        vacancy_energies,
                        vacancy_pdos,
                    )
                )

                visible_max = max(
                    visible_max,
                    pristine_visible_max,
                    vacancy_visible_max,
                )

                # --------------------------------------------
                # Plot pristine and vacancy PDOS
                # --------------------------------------------

                ax.plot(
                    pristine_energies,
                    pristine_pdos,
                    linestyle="--",
                    linewidth=1.3,
                    label=(
                        f"Pristine "
                        f"{orbital_name}"
                    ),
                )

                ax.plot(
                    vacancy_energies,
                    vacancy_pdos,
                    linestyle="-",
                    linewidth=1.3,
                    label=(
                        f"Vacancy "
                        f"{orbital_name}"
                    ),
                )

                # --------------------------------------------
                # Save pristine data
                # --------------------------------------------

                for energy, value in zip(
                    pristine_energies,
                    pristine_pdos,
                ):

                    output_rows.append(
                        {
                            "structure":
                                structure_name,
                            "coordination_type":
                                coordination_type,
                            "reference_ga_atom_number":
                                reference_atom_number,
                            "defect_ga_atom_number":
                                defect_atom_number,
                            "system":
                                "pristine",
                            "orbital":
                                orbital_name,
                            "energy_minus_fermi_eV":
                                energy,
                            "pdos_states_per_eV":
                                value,
                        }
                    )

                # --------------------------------------------
                # Save vacancy data
                # --------------------------------------------

                for energy, value in zip(
                    vacancy_energies,
                    vacancy_pdos,
                ):

                    output_rows.append(
                        {
                            "structure":
                                structure_name,
                            "coordination_type":
                                coordination_type,
                            "reference_ga_atom_number":
                                reference_atom_number,
                            "defect_ga_atom_number":
                                defect_atom_number,
                            "system":
                                "vacancy",
                            "orbital":
                                orbital_name,
                            "energy_minus_fermi_eV":
                                energy,
                            "pdos_states_per_eV":
                                value,
                        }
                    )

            # ------------------------------------------------
            # Plot formatting
            # ------------------------------------------------

            ax.axvline(
                0.0,
                linestyle=":",
                linewidth=1.0,
                label=r"$E_F$",
            )

            ax.set_xlim(
                ENERGY_MIN_EV,
                ENERGY_MAX_EV,
            )

            if visible_max > 0:

                ax.set_ylim(
                    0.0,
                    visible_max
                    * Y_AXIS_MARGIN,
                )

            ax.set_ylabel(
                "Ga PDOS\n(states/eV)"
            )

            ax.set_title(
                f"Pristine Ga"
                f"{reference_atom_number} "
                f"vs defect Ga"
                f"{defect_atom_number} "
                f"({coordination_type})"
            )

            ax.grid(
                alpha=0.2
            )

            ax.legend(
                fontsize=8,
                ncol=2,
            )

        # ----------------------------------------------------
        # Figure formatting
        # ----------------------------------------------------

        axes[-1].set_xlabel(
            r"Energy, $E-E_F$ (eV)"
        )

        fig.suptitle(
            "Matched first-shell Ga PDOS: "
            f"{VACANCY_LABELS[structure_name]}",
            fontsize=14,
        )

        fig.tight_layout(
            rect=[
                0,
                0,
                1,
                0.97,
            ]
        )

        # ----------------------------------------------------
        # Save figure
        # ----------------------------------------------------

        figure_path = (
            FIGURE_DIR
            / (
                f"{structure_name}_"
                "ga_pdos_pristine_"
                "comparison.png"
            )
        )

        fig.savefig(
            figure_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ----------------------------------------------------
        # Save CSV
        # ----------------------------------------------------

        output_df = pd.DataFrame(
            output_rows
        )

        csv_path = (
            COMPARISON_OUTPUT_DIR
            / (
                f"{structure_name}_"
                "ga_pdos_comparison.csv"
            )
        )

        output_df.to_csv(
            csv_path,
            index=False,
        )

        print()
        print("Saved CSV:")
        print(csv_path)

        print("Saved figure:")
        print(figure_path)

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("Finished.")
    print("=" * 72)


if __name__ == "__main__":
    main()