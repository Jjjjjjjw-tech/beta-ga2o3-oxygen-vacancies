"""
Electronic-structure dataset verification.

This script checks whether the newly imported VASP SCF dataset
contains usable DOS and projected DOS information for:

- pristine
- vacancy O1
- vacancy O2
- vacancy O3

All data paths are read from the project-level config.py.
"""

from pathlib import Path
import sys

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
)


# ============================================================
# Dataset definition
# ============================================================

CALCULATIONS = {
    "pristine": PRISTINE_SCF_DIR,
    "vacancy_O1": VACANCY_O1_SCF_DIR,
    "vacancy_O2": VACANCY_O2_SCF_DIR,
    "vacancy_O3": VACANCY_O3_SCF_DIR,
}


# ============================================================
# Files required for electronic-structure analysis
# ============================================================

FILES_TO_CHECK = [
    "INCAR",
    "OUTCAR",
    "vasprun.xml",
    "DOSCAR",
    "CHGCAR",
    "AECCAR0",
    "AECCAR1",
    "AECCAR2",
    "ELFCAR",
    "LOCPOT",
    "WAVECAR",
]


# ============================================================
# Helper functions
# ============================================================

def print_header(title: str) -> None:
    """Print a formatted section header."""

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def check_files(folder: Path) -> dict[str, bool]:
    """
    Check whether required VASP files exist.

    Returns
    -------
    dict
        Mapping from file name to existence status.
    """

    return {
        file_name: (folder / file_name).exists()
        for file_name in FILES_TO_CHECK
    }


def analyze_vasprun(
    name: str,
    folder: Path,
) -> dict:
    """
    Read vasprun.xml and summarize DOS/PDOS information.

    Parameters
    ----------
    name
        Human-readable calculation label.

    folder
        SCF calculation directory.

    Returns
    -------
    dict
        Summary of electronic-structure information.
    """

    vasprun_path = folder / "vasprun.xml"

    if not vasprun_path.exists():
        raise FileNotFoundError(
            f"{name}: vasprun.xml not found:\n"
            f"{vasprun_path}"
        )

    print(f"Reading {name} ...")

    vasprun = Vasprun(
        vasprun_path,
        parse_dos=True,
        parse_eigen=True,
        parse_projected_eigen=False,
    )

    complete_dos = vasprun.complete_dos

    pdos_entries = len(
        complete_dos.pdos
    )

    number_of_atoms = len(
        vasprun.final_structure
    )

    band_gap = complete_dos.get_gap()

    efermi = float(
        vasprun.efermi
    )

    return {
        "name": name,
        "folder": folder,
        "number_of_atoms": number_of_atoms,
        "electronic_converged":
            vasprun.converged_electronic,
        "ionic_converged":
            vasprun.converged_ionic,
        "overall_converged":
            vasprun.converged,
        "efermi_eV": efermi,
        "band_gap_eV": band_gap,
        "pdos_entries": pdos_entries,
        "has_pdos": pdos_entries > 0,
        "complete_dos": complete_dos,
    }


def print_file_status(
    name: str,
    folder: Path,
) -> None:
    """Print required-file availability."""

    print_header(
        f"{name} — file check"
    )

    print(
        f"Directory:\n"
        f"{folder}"
    )

    print()

    file_status = check_files(
        folder
    )

    for file_name, exists in file_status.items():
        status = (
            "FOUND"
            if exists
            else "MISSING"
        )

        print(
            f"{file_name:15s}: "
            f"{status}"
        )


def print_dos_summary(
    result: dict,
) -> None:
    """Print DOS and PDOS verification results."""

    name = result["name"]

    print_header(
        f"{name} — DOS / PDOS verification"
    )

    print(
        f"Number of atoms       : "
        f"{result['number_of_atoms']}"
    )

    print(
        f"Electronic converged  : "
        f"{result['electronic_converged']}"
    )

    print(
        f"Ionic converged       : "
        f"{result['ionic_converged']}"
    )

    print(
        f"Overall converged     : "
        f"{result['overall_converged']}"
    )

    print(
        f"Fermi energy          : "
        f"{result['efermi_eV']:.6f} eV"
    )

    print(
        f"DOS band gap          : "
        f"{result['band_gap_eV']:.6f} eV"
    )

    print(
        f"PDOS entries          : "
        f"{result['pdos_entries']}"
    )

    print(
        f"Projected DOS usable  : "
        f"{result['has_pdos']}"
    )

    if result["has_pdos"]:

        pdos = (
            result["complete_dos"]
            .pdos
        )

        first_site = next(
            iter(pdos)
        )

        orbitals = list(
            pdos[first_site].keys()
        )

        print()

        print(
            "Example projected site:"
        )

        print(
            f"  {first_site}"
        )

        print()

        print(
            "Available orbitals:"
        )

        for orbital in orbitals:
            print(
                f"  {orbital}"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print_header(
        "Electronic-structure dataset verification"
    )

    print(
        "This script verifies the newly imported "
        "VASP SCF dataset."
    )

    print()

    print(
        f"Project root:\n"
        f"{PROJECT_ROOT}"
    )

    results = {}

    # --------------------------------------------------------
    # File checks
    # --------------------------------------------------------

    for name, folder in CALCULATIONS.items():

        print_file_status(
            name,
            folder,
        )

    # --------------------------------------------------------
    # DOS / PDOS checks
    # --------------------------------------------------------

    for name, folder in CALCULATIONS.items():

        try:

            result = analyze_vasprun(
                name,
                folder,
            )

            results[name] = result

            print_dos_summary(
                result
            )

        except Exception as error:

            print_header(
                f"{name} — ERROR"
            )

            print(
                type(error).__name__
            )

            print(
                error
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_header(
        "Final dataset summary"
    )

    for name in CALCULATIONS:

        if name not in results:

            print(
                f"{name:12s}: "
                f"FAILED"
            )

            continue

        result = results[name]

        print(
            f"{name:12s}: "
            f"atoms={result['number_of_atoms']:3d} | "
            f"converged={str(result['overall_converged']):5s} | "
            f"PDOS={result['pdos_entries']:3d} | "
            f"gap={result['band_gap_eV']:.4f} eV"
        )

    print()

    print(
        "Expected PDOS entry count:"
    )

    print(
        "  pristine   -> approximately 80"
    )

    print(
        "  vacancies  -> approximately 79"
    )

    print()

    print(
        "If PDOS entries are greater than zero for "
        "all four systems, the dataset is ready for "
        "site- and orbital-projected DOS analysis."
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()