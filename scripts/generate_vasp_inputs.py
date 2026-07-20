from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar, Kpoints

project_root = Path(__file__).resolve().parent.parent
structures_dir = project_root /"structures"
output_dir = project_root / "vasp_calculations"
output_dir.mkdir(parents=True, exist_ok=True)

def write_poscar(structure, calculation_dir):
    poscar = Poscar(structure)
    poscar.write_file(calculation_dir / "POSCAR")

def write_incar(calculation_dir):
    incar_content = """SYSTEM = beta-Ga2O3 oxygen vacancy

    IBRION = 2
    EDIFFG = -0.02
    """

    incar_file = calculation_dir / "INCAR"
    incar_file.write_text(incar_content)

def write_kpoints(calculation_dir):
    kpoints = Kpoints.gamma_automatic((4, 4, 4))
    kpoints.write_file(calculation_dir / "KPOINTS")

vacancy_files = structures_dir.glob("*vacancy*.cif")
for vacancy_file in vacancy_files:
    structure = Structure.from_file(vacancy_file)
    
    parts = vacancy_file.stem.split("_")
    folder_name = f"{parts[-2]}_{parts[-1]}"
    calculation_dir = output_dir / folder_name  
    calculation_dir.mkdir(parents=True, exist_ok=True)

    write_poscar(structure, calculation_dir)    
    write_incar(calculation_dir)
    write_kpoints(calculation_dir)

    
