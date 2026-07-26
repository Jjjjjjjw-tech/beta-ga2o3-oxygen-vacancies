from pathlib import Path

import numpy as np
from pymatgen.core import Structure

BASE = Path("beta-Ga2O3_oxygen_vacancy_VASP_inputs/received_results/2026-07-25_results")

pristine_path = BASE / "pristine" / "relax" / "CONTCAR"
vacancy_path = BASE / "vacancy_O1" / "relax" / "POSCAR"

pristine = Structure.from_file(pristine_path)
vacancy = Structure.from_file(vacancy_path)

print(f"Pristine atoms: {len(pristine)}")
print(f"Vacancy atoms : {len(vacancy)}")

# 先检查晶格是否一致
lattice_diff = np.max(
    np.abs(pristine.lattice.matrix - vacancy.lattice.matrix)
)
print(f"Maximum lattice difference: {lattice_diff:.8e} Å")

# 对 vacancy 中的每个原子，在 pristine 中寻找：
# 1. 元素相同
# 2. 周期性边界条件下距离最近的原子
used_pristine_indices = set()
matches = []

for vacancy_index, vacancy_site in enumerate(vacancy):
    candidates = []

    for pristine_index, pristine_site in enumerate(pristine):
        if pristine_index in used_pristine_indices:
            continue

        if pristine_site.specie != vacancy_site.specie:
            continue

        distance = pristine.lattice.get_distance_and_image(
            pristine_site.frac_coords,
            vacancy_site.frac_coords,
        )[0]

        candidates.append((distance, pristine_index))

    if not candidates:
        raise RuntimeError(
            f"No matching pristine atom found for vacancy atom {vacancy_index + 1}"
        )

    distance, pristine_index = min(candidates)
    used_pristine_indices.add(pristine_index)
    matches.append((vacancy_index, pristine_index, distance))

max_distance = max(distance for _, _, distance in matches)

missing_indices = [
    index for index in range(len(pristine))
    if index not in used_pristine_indices
]

print(f"Maximum matching distance: {max_distance:.8e} Å")
print(f"Unmatched pristine atoms: {len(missing_indices)}")

for index in missing_indices:
    site = pristine[index]
    print(
        f"Missing atom: pristine index {index + 1}, "
        f"element {site.specie}, "
        f"fractional coordinates {site.frac_coords}"
    )

tolerance = 1e-5

if (
    lattice_diff < tolerance
    and max_distance < tolerance
    and len(missing_indices) == 1
    and pristine[missing_indices[0]].specie.symbol == "O"
):
    print("\nRESULT: Vacancy_O1 POSCAR was generated from the relaxed pristine")
    print("structure by deleting exactly one oxygen atom.")
else:
    print("\nRESULT: The structures are not related by simply deleting one oxygen.")