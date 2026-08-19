#!/usr/bin/env python3

"""
Explore the internal structure of projected DOS (PDOS) data.

Purpose
-------
This script does NOT perform scientific analysis yet.

It is only used to understand how pymatgen stores:

- atoms / sites
- orbitals
- spin channels
- energy grid
- orbital-resolved DOS arrays

The script reads the pristine calculation from config.py.
"""

from pathlib import Path
import sys

import numpy as np

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
)


# ============================================================
# Input
# ============================================================

VASPRUN_PATH = (
    PRISTINE_SCF_DIR
    / "vasprun.xml"
)


# ============================================================
# Helper
# ============================================================

def print_header(title: str) -> None:
    """Print a formatted section header."""

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# Main
# ============================================================

def main() -> None:

    print_header(
        "Explore projected DOS data structure"
    )

    print(
        f"vasprun.xml:\n"
        f"{VASPRUN_PATH}"
    )

    if not VASPRUN_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find:\n{VASPRUN_PATH}"
        )

    # --------------------------------------------------------
    # Read vasprun.xml
    # --------------------------------------------------------

    print()
    print("Reading vasprun.xml ...")

    vasprun = Vasprun(
        VASPRUN_PATH,
        parse_dos=True,
        parse_eigen=False,
        parse_projected_eigen=False,
    )

    complete_dos = (
        vasprun.complete_dos
    )

    pdos = (
        complete_dos.pdos
    )

    structure = (
        complete_dos.structure
    )

    energies = np.asarray(
        complete_dos.energies,
        dtype=float,
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print_header(
        "1. Basic information"
    )

    print(
        f"Type of complete_dos : "
        f"{type(complete_dos)}"
    )

    print(
        f"Type of pdos         : "
        f"{type(pdos)}"
    )

    print(
        f"Number of atoms      : "
        f"{len(structure)}"
    )

    print(
        f"Number of PDOS sites : "
        f"{len(pdos)}"
    )

    print(
        f"Energy points        : "
        f"{len(energies)}"
    )

    print(
        f"Fermi energy         : "
        f"{complete_dos.efermi:.6f} eV"
    )

    print(
        f"Energy minimum       : "
        f"{energies.min():.6f} eV"
    )

    print(
        f"Energy maximum       : "
        f"{energies.max():.6f} eV"
    )

    # --------------------------------------------------------
    # Inspect first few sites
    # --------------------------------------------------------

    print_header(
        "2. First five sites"
    )

    for index, site in enumerate(
        structure[:5]
    ):

        print(
            f"Python index {index:2d} | "
            f"Atom number {index + 1:2d} | "
            f"{site.species_string}"
        )

        print(
            f"  Fractional coordinates: "
            f"{site.frac_coords}"
        )

    # --------------------------------------------------------
    # Inspect one PDOS site
    # --------------------------------------------------------

    print_header(
        "3. PDOS of the first atom"
    )

    first_site = structure[0]

    print(
        f"Site object:\n"
        f"{first_site}"
    )

    print()

    print(
        f"Element: "
        f"{first_site.species_string}"
    )

    print()

    first_site_pdos = (
        pdos[first_site]
    )

    print(
        f"Type of site PDOS: "
        f"{type(first_site_pdos)}"
    )

    print()

    print(
        "Available orbital keys:"
    )

    for orbital in first_site_pdos.keys():
        print(
            f"  {orbital}"
        )

    # --------------------------------------------------------
    # Inspect one orbital
    # --------------------------------------------------------

    print_header(
        "4. Inspect the first orbital"
    )

    first_orbital = next(
        iter(first_site_pdos)
    )

    orbital_spin_data = (
        first_site_pdos[
            first_orbital
        ]
    )

    print(
        f"Selected orbital      : "
        f"{first_orbital}"
    )

    print(
        f"Type of orbital data  : "
        f"{type(orbital_spin_data)}"
    )

    print()

    print(
        "Available spin channels:"
    )

    for spin in orbital_spin_data.keys():
        print(
            f"  {spin}"
        )

    # --------------------------------------------------------
    # Inspect DOS array
    # --------------------------------------------------------

    print_header(
        "5. DOS array for the first orbital"
    )

    for spin, density in (
        orbital_spin_data.items()
    ):

        density_array = np.asarray(
            density,
            dtype=float,
        )

        print(
            f"Spin channel: {spin}"
        )

        print(
            f"  Array type : "
            f"{type(density_array)}"
        )

        print(
            f"  Shape      : "
            f"{density_array.shape}"
        )

        print(
            f"  Minimum    : "
            f"{density_array.min():.6f}"
        )

        print(
            f"  Maximum    : "
            f"{density_array.max():.6f}"
        )

        print(
            f"  First 10 values:"
        )

        print(
            density_array[:10]
        )

    # --------------------------------------------------------
    # Summarize orbital groups
    # --------------------------------------------------------

    print_header(
        "6. Orbital groups"
    )

    orbital_names = [
        str(orbital)
        for orbital
        in first_site_pdos.keys()
    ]

    s_orbitals = [
        orbital
        for orbital in orbital_names
        if orbital == "s"
    ]

    p_orbitals = [
        orbital
        for orbital in orbital_names
        if orbital in (
            "px",
            "py",
            "pz",
        )
    ]

    d_orbitals = [
        orbital
        for orbital in orbital_names
        if orbital.startswith("d")
    ]

    print(
        f"s orbitals: "
        f"{s_orbitals}"
    )

    print(
        f"p orbitals: "
        f"{p_orbitals}"
    )

    print(
        f"d orbitals: "
        f"{d_orbitals}"
    )

    # --------------------------------------------------------
    # Final conceptual summary
    # --------------------------------------------------------

    print_header(
        "7. PDOS data hierarchy"
    )

    print(
        "complete_dos.pdos"
    )

    print(
        "    ↓"
    )

    print(
        "site / atom"
    )

    print(
        "    ↓"
    )

    print(
        "orbital"
    )

    print(
        "    ↓"
    )

    print(
        "spin channel"
    )

    print(
        "    ↓"
    )

    print(
        "DOS array over the energy grid"
    )


if __name__ == "__main__":
    main()