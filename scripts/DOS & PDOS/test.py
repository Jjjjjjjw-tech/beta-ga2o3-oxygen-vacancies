from pathlib import Path
from pymatgen.io.vasp.outputs import Vasprun

path = Path(
    "beta-Ga2O3_oxygen_vacancy_VASP_inputs"
    "/received_results/2026-07-25_results/pristine/scf/vasprun.xml"
)

vasprun = Vasprun(path)

print(type(vasprun.complete_dos))

print()

print("pdos dict length:")
print(len(vasprun.complete_dos.pdos))

print()

print("keys:")
print(list(vasprun.complete_dos.pdos.keys())[:5])