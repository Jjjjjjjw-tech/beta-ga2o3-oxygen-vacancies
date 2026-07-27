from pathlib import Path

from pymatgen.io.vasp.outputs import Vasprun


BASE = Path(
    "beta-Ga2O3_oxygen_vacancy_VASP_inputs"
    "/received_results/2026-07-25_results"
)

calculations = {
    "pristine": BASE / "pristine" / "scf" / "vasprun.xml",
    "vacancy_O1": BASE / "vacancy_O1" / "scf" / "vasprun.xml",
    "vacancy_O2": BASE / "vacancy_O2" / "scf" / "vasprun.xml",
    "vacancy_O3": BASE / "vacancy_O3" / "scf" / "vasprun.xml",
}

energies = {}

for name, path in calculations.items():
    if not path.exists():
        print(f"{name}: 文件不存在")
        print(f"  {path}")
        continue

    vasprun = Vasprun(
        path,
        parse_dos=False,
        parse_eigen=False,
        parse_projected_eigen=False,
    )

    energy = float(vasprun.final_energy)
    energies[name] = energy

    print(f"{name}")
    print(f"  Final energy         : {energy:.8f} eV")
    print(f"  Electronic converged : {vasprun.converged_electronic}")
    print(f"  Overall converged    : {vasprun.converged}")
    print()

if "pristine" in energies:
    print("Energy difference relative to pristine:")
    for name in ("vacancy_O1", "vacancy_O2", "vacancy_O3"):
        if name in energies:
            delta_e = energies[name] - energies["pristine"]
            print(f"  {name:12s}: {delta_e:.8f} eV")

vacancy_names = [
    name for name in ("vacancy_O1", "vacancy_O2", "vacancy_O3")
    if name in energies
]

if len(vacancy_names) == 3:
    print("\nRelative stability among vacancy structures:")
    ordered = sorted(vacancy_names, key=lambda name: energies[name])

    lowest_energy = energies[ordered[0]]

    for rank, name in enumerate(ordered, start=1):
        relative_energy = energies[name] - lowest_energy
        print(
            f"  {rank}. {name:12s} "
            f"{energies[name]:.8f} eV "
            f"(ΔE = {relative_energy:.8f} eV)"
        )