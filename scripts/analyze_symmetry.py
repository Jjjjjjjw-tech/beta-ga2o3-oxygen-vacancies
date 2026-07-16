from pathlib import Path

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


cif_path = Path("structures") / "beta-Ga2O3_mp-886.cif"
structure = Structure.from_file(cif_path)

analyzer = SpacegroupAnalyzer(
    structure,
    symprec=1e-3,
    angle_tolerance=5,
)

print("Space group symbol:", analyzer.get_space_group_symbol())
print("Space group number:", analyzer.get_space_group_number())
symmetry_dataset = analyzer.get_symmetry_dataset()

equivalent_atoms = symmetry_dataset.equivalent_atoms

print("\nEquivalent-atom labels:")
for index, label in enumerate(equivalent_atoms):
    element = structure[index].species_string
    print(f"Index {index:>2}  Element {element:<2}  Group {label}")

oxygen_groups = {}

for index, site in enumerate(structure):
    if site.species_string != "O":
        continue

    group_label = int(equivalent_atoms[index])

    if group_label not in oxygen_groups:
        oxygen_groups[group_label] = []

    oxygen_groups[group_label].append(index)

print("\nSymmetry-equivalent oxygen groups:")
for group_number, indices in enumerate(oxygen_groups.values(), start=1):
    print(f"Oxygen type {group_number}: indices {indices}")