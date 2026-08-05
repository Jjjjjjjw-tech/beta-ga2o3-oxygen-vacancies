#!/usr/bin/env python3
"""
Extract final SCF energies from VASP vasprun.xml files.

Outputs
-------
analysis/energy/total_energies.csv
analysis/energy/vacancy_relative_energies.csv

Notes
-----
The vacancy-relative energies compare vacancy_O1, vacancy_O2,
and vacancy_O3 directly.

Because all three vacancy structures contain the same number of atoms
and each removes one oxygen atom, their relative total-energy ordering
also gives their relative vacancy stability under the same calculation
settings.

However:

    E_vacancy - E_pristine

is not yet the absolute oxygen-vacancy formation energy. The oxygen
chemical potential must still be added later.
"""

from __future__ import annotations

import csv
from pathlib import Path

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "analysis"
    / "energy"
)

TOTAL_ENERGY_OUTPUT_PATH = (
    OUTPUT_DIR
    / "total_energies.csv"
)

RELATIVE_ENERGY_OUTPUT_PATH = (
    OUTPUT_DIR
    / "vacancy_relative_energies.csv"
)


# ============================================================
# Build calculation paths
# ============================================================

CALCULATIONS = {
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
    """Write a list of dictionaries to a CSV file."""

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


def extract_vasprun_energy(
    structure_name: str,
    vasprun_path: Path,
) -> dict[str, object]:
    """
    Extract final energy and convergence information from vasprun.xml.
    """

    if not vasprun_path.is_file():
        raise FileNotFoundError(
            f"{structure_name}: cannot find vasprun.xml:\n"
            f"{vasprun_path}"
        )

    vasprun = Vasprun(
        vasprun_path,
        parse_dos=False,
        parse_eigen=False,
        parse_projected_eigen=False,
    )

    return {
        "structure": structure_name,
        "vasprun_path": str(vasprun_path),
        "total_energy_eV": float(
            vasprun.final_energy
        ),
        "electronic_converged": bool(
            vasprun.converged_electronic
        ),
        "ionic_converged": bool(
            vasprun.converged_ionic
        ),
        "overall_converged": bool(
            vasprun.converged
        ),
    }


# ============================================================
# Energy analysis
# ============================================================

def extract_all_energies() -> list[dict[str, object]]:
    """Extract energies for all configured structures."""

    energy_rows: list[dict[str, object]] = []

    print("=" * 78)
    print("SCF total energies")
    print("=" * 78)

    for structure_name in STRUCTURE_NAMES:
        vasprun_path = CALCULATIONS[
            structure_name
        ]

        try:
            result = extract_vasprun_energy(
                structure_name=structure_name,
                vasprun_path=vasprun_path,
            )

        except FileNotFoundError as error:
            print()
            print(error)
            continue

        energy_rows.append(result)

        print()
        print(structure_name)
        print(
            "  Final energy         : "
            f"{result['total_energy_eV']:.8f} eV"
        )
        print(
            "  Electronic converged : "
            f"{result['electronic_converged']}"
        )
        print(
            "  Ionic converged      : "
            f"{result['ionic_converged']}"
        )
        print(
            "  Overall converged    : "
            f"{result['overall_converged']}"
        )

    if not energy_rows:
        raise RuntimeError(
            "No energies could be extracted."
        )

    return energy_rows


def build_energy_dictionary(
    energy_rows: list[dict[str, object]],
) -> dict[str, float]:
    """Build structure-name to total-energy mapping."""

    return {
        str(row["structure"]):
            float(row["total_energy_eV"])
        for row in energy_rows
    }


def print_energy_differences_to_pristine(
    energies: dict[str, float],
) -> None:
    """
    Print E(vacancy) - E(pristine).

    This value is not yet the absolute formation energy.
    """

    if "pristine" not in energies:
        print()
        print(
            "Cannot calculate energy differences to pristine: "
            "pristine energy is missing."
        )
        return

    pristine_energy = energies[
        "pristine"
    ]

    print()
    print("=" * 78)
    print("Energy differences relative to pristine")
    print("=" * 78)

    for structure_name in (
        "vacancy_O1",
        "vacancy_O2",
        "vacancy_O3",
    ):
        if structure_name not in energies:
            continue

        delta_energy = (
            energies[structure_name]
            - pristine_energy
        )

        print(
            f"{structure_name:12s}: "
            f"{delta_energy:+.8f} eV"
        )

    print()
    print(
        "Note: these values are not absolute vacancy "
        "formation energies because the oxygen chemical "
        "potential has not yet been included."
    )


def build_relative_vacancy_rows(
    energies: dict[str, float],
) -> list[dict[str, object]]:
    """
    Rank vacancy structures relative to the lowest-energy vacancy.
    """

    vacancy_names = [
        structure_name
        for structure_name in (
            "vacancy_O1",
            "vacancy_O2",
            "vacancy_O3",
        )
        if structure_name in energies
    ]

    if not vacancy_names:
        return []

    ordered_names = sorted(
        vacancy_names,
        key=lambda name: energies[name],
    )

    lowest_energy = energies[
        ordered_names[0]
    ]

    relative_rows: list[
        dict[str, object]
    ] = []

    for rank, structure_name in enumerate(
        ordered_names,
        start=1,
    ):
        total_energy = energies[
            structure_name
        ]

        relative_energy = (
            total_energy - lowest_energy
        )

        relative_rows.append(
            {
                "rank": rank,
                "structure": structure_name,
                "total_energy_eV":
                    total_energy,
                "relative_energy_eV":
                    relative_energy,
            }
        )

    return relative_rows


def print_relative_vacancy_stability(
    relative_rows: list[dict[str, object]],
) -> None:
    """Print vacancy structures ranked by total energy."""

    if not relative_rows:
        print()
        print(
            "No vacancy structures are available "
            "for relative-energy ranking."
        )
        return

    print()
    print("=" * 78)
    print("Relative stability among vacancy structures")
    print("=" * 78)

    for row in relative_rows:
        print(
            f"{int(row['rank'])}. "
            f"{str(row['structure']):12s} "
            f"{float(row['total_energy_eV']):.8f} eV "
            f"(relative energy = "
            f"{float(row['relative_energy_eV']):.8f} eV)"
        )


# ============================================================
# Main program
# ============================================================

def main() -> None:
    """Extract, analyze, and save SCF total energies."""

    energy_rows = extract_all_energies()

    energies = build_energy_dictionary(
        energy_rows
    )

    print_energy_differences_to_pristine(
        energies
    )

    relative_rows = (
        build_relative_vacancy_rows(
            energies
        )
    )

    print_relative_vacancy_stability(
        relative_rows
    )

    write_csv(
        output_path=TOTAL_ENERGY_OUTPUT_PATH,
        rows=energy_rows,
    )

    if relative_rows:
        write_csv(
            output_path=
                RELATIVE_ENERGY_OUTPUT_PATH,
            rows=relative_rows,
        )

    print()
    print("=" * 78)
    print("Output files")
    print("=" * 78)

    print(TOTAL_ENERGY_OUTPUT_PATH)

    if relative_rows:
        print(
            RELATIVE_ENERGY_OUTPUT_PATH
        )


if __name__ == "__main__":
    main()