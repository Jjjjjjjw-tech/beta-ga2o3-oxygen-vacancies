from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp import Kpoints, Poscar


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRUCTURES_DIR = PROJECT_ROOT / "structures" / "supercell"
OUTPUT_DIR = PROJECT_ROOT / "vasp_calculations"

STRUCTURE_FILES = {
    "pristine": "beta-Ga2O3_pristine_2x2x2.cif",
    "vacancy_O1": "beta-Ga2O3_vacancy_O1_2x2x2.cif",
    "vacancy_O2": "beta-Ga2O3_vacancy_O2_2x2x2.cif",
    "vacancy_O3": "beta-Ga2O3_vacancy_O3_2x2x2.cif",
}


def write_poscar(
    structure: Structure,
    calculation_dir: Path,
) -> None:
    """Write a VASP POSCAR file."""
    Poscar(structure).write_file(calculation_dir / "POSCAR")


def write_relax_incar(
    calculation_name: str,
    calculation_dir: Path,
) -> None:
    """Write the INCAR for geometry relaxation."""

    # Pristine bulk: relax ions, cell shape and volume.
    # Vacancy structures: keep the bulk lattice fixed and relax ions only.
    isif = 3 if calculation_name == "pristine" else 2

    incar_content = f"""SYSTEM = beta-Ga2O3 {calculation_name} relaxation

# Electronic calculation
PREC    = Accurate
ENCUT   = 520
EDIFF   = 1E-6
ALGO    = Normal
NELM    = 120
NELMIN  = 6

# Ionic relaxation
IBRION  = 2
NSW     = 200
ISIF    = {isif}
EDIFFG  = -0.02
POTIM   = 0.5

# Occupation settings
ISMEAR  = 0
SIGMA   = 0.05

# Spin and symmetry
ISPIN   = 2
ISYM    = 0

# Accuracy settings
LREAL   = .FALSE.
LASPH   = .TRUE.
ADDGRID = .TRUE.
LMAXMIX = 4

# Output settings
LWAVE   = .FALSE.
LCHARG  = .FALSE.
"""

    (calculation_dir / "INCAR").write_text(
        incar_content,
        encoding="utf-8",
    )


def write_static_incar(
    calculation_name: str,
    calculation_dir: Path,
) -> None:
    """Write the INCAR for the final static total-energy calculation."""

    incar_content = f"""SYSTEM = beta-Ga2O3 {calculation_name} static

# Electronic calculation
PREC    = Accurate
ENCUT   = 520
EDIFF   = 1E-7
ALGO    = Normal
NELM    = 150
NELMIN  = 6

# Static calculation: atoms and lattice do not move
IBRION  = -1
NSW     = 0

# Occupation settings
ISMEAR  = 0
SIGMA   = 0.05

# Spin and symmetry
ISPIN   = 2
ISYM    = 0

# Accuracy settings
LREAL   = .FALSE.
LASPH   = .TRUE.
ADDGRID = .TRUE.
LMAXMIX = 4

# Output settings
LWAVE   = .FALSE.
LCHARG  = .TRUE.
"""

    (calculation_dir / "INCAR").write_text(
        incar_content,
        encoding="utf-8",
    )


def write_kpoints(calculation_dir: Path) -> None:
    """Write a Gamma-centered 2 x 2 x 2 k-point mesh."""
    kpoints = Kpoints.gamma_automatic(kpts=(2, 2, 2))
    kpoints.write_file(calculation_dir / "KPOINTS")


def write_readme(
    calculation_name: str,
    relax_dir: Path,
    static_dir: Path,
) -> None:
    """Write calculation instructions for the VASP operator."""

    readme_content = f"""Calculation: beta-Ga2O3 {calculation_name}
Charge state: Neutral (q = 0)

========================================
Pseudopotentials
========================================

Please use the standard VASP PAW-PBE pseudopotentials:

- Ga_d
- O

The POTCAR order must match the POSCAR:

Ga
O

========================================
Step 1: Geometry Relaxation
========================================

Run the calculation in:

relax/{calculation_name}

========================================
Step 2: Static Calculation
========================================

After the relaxation is finished:

1. Copy:
   relax/{calculation_name}/CONTCAR

2. Paste it into:
   static/{calculation_name}/

3. Rename:
   CONTCAR -> POSCAR

4. Run the static calculation.

========================================
Please return the following files
========================================

Relaxation:

- INCAR
- POSCAR
- CONTCAR
- KPOINTS
- OUTCAR
- OSZICAR
- vasprun.xml

Static:

- INCAR
- POSCAR
- KPOINTS
- OUTCAR
- OSZICAR
- vasprun.xml

========================================
Please also report
========================================

- Final total energy (TOTEN)
- VASP version
- POTCAR TITEL entries
- Whether both electronic and ionic convergence were achieved

-If any calculation fails or does not converge,
-please also return all generated output files.

Thank you very much!
"""

    (relax_dir / "README.txt").write_text(
        readme_content,
        encoding="utf-8",
    )

    (static_dir / "README.txt").write_text(
        readme_content,
        encoding="utf-8",
    )


def validate_structure(
    calculation_name: str,
    structure: Structure,
) -> None:
    """Validate atom count and composition."""

    expected_atoms = 80 if calculation_name == "pristine" else 79
    expected_oxygen = 48 if calculation_name == "pristine" else 47

    ga_count = int(structure.composition["Ga"])
    o_count = int(structure.composition["O"])

    if len(structure) != expected_atoms:
        raise ValueError(
            f"{calculation_name}: expected {expected_atoms} atoms, "
            f"but found {len(structure)}."
        )

    if ga_count != 32 or o_count != expected_oxygen:
        raise ValueError(
            f"{calculation_name}: unexpected composition "
            f"Ga{ga_count}O{o_count}."
        )


def main() -> None:
    relax_root = OUTPUT_DIR / "relax"
    static_root = OUTPUT_DIR / "static"

    relax_root.mkdir(parents=True, exist_ok=True)
    static_root.mkdir(parents=True, exist_ok=True)

    for calculation_name, filename in STRUCTURE_FILES.items():
        structure_path = STRUCTURES_DIR / filename

        if not structure_path.exists():
            raise FileNotFoundError(
                f"Structure file not found: {structure_path}"
            )

        structure = Structure.from_file(structure_path)
        validate_structure(calculation_name, structure)

        relax_dir = relax_root / calculation_name
        static_dir = static_root / calculation_name

        relax_dir.mkdir(parents=True, exist_ok=True)
        static_dir.mkdir(parents=True, exist_ok=True)

        # Relaxation inputs
        write_poscar(structure, relax_dir)
        write_relax_incar(calculation_name, relax_dir)
        write_kpoints(relax_dir)

        # Static inputs
        # This POSCAR is only a placeholder.
        # It must be replaced by the relaxed CONTCAR before static calculation.
        write_poscar(structure, static_dir)
        write_static_incar(calculation_name, static_dir)
        write_kpoints(static_dir)

        write_readme(
            calculation_name=calculation_name,
            relax_dir=relax_dir,
            static_dir=static_dir,
        )

        print(
            f"{calculation_name}: "
            f"{structure.composition}, "
            f"{len(structure)} atoms"
        )
        print(f"Relax inputs:  {relax_dir}")
        print(f"Static inputs: {static_dir}")

    print("\nRelaxation and static input files generated successfully.")
    print(
        "Important: replace every static POSCAR with the corresponding "
        "relaxed CONTCAR before running the static calculations."
    )


if __name__ == "__main__":
    main()