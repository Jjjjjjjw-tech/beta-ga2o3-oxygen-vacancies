from collections import Counter
from pathlib import Path

from pymatgen.core import Structure


def main() -> None:
    cif_path = Path("structures") / "beta-Ga2O3_mp-886.cif"

    if not cif_path.exists():
        raise FileNotFoundError(f"Cannot find CIF file: {cif_path}")

    structure = Structure.from_file(cif_path)

    print("=" * 60)
    print("Basic structure information")
    print("=" * 60)

    print(f"Formula: {structure.composition.reduced_formula}")
    print(f"Number of atoms: {len(structure)}")
    print(f"Volume: {structure.volume:.3f} Å³")

    print("\nLattice parameters")
    print(f"a = {structure.lattice.a:.4f} Å")
    print(f"b = {structure.lattice.b:.4f} Å")
    print(f"c = {structure.lattice.c:.4f} Å")
    print(f"alpha = {structure.lattice.alpha:.4f}°")
    print(f"beta  = {structure.lattice.beta:.4f}°")
    print(f"gamma = {structure.lattice.gamma:.4f}°")

    element_counts = Counter(site.species_string for site in structure)

    print("\nElement counts")
    for element, count in element_counts.items():
        print(f"{element}: {count}")

    print("\nAtomic sites")
    print("Index  Element        Fractional coordinates")
    print("-" * 60)

    for index, site in enumerate(structure):
        x, y, z = site.frac_coords
        print(
            f"{index:>5}  "
            f"{site.species_string:<10}  "
            f"({x:>9.6f}, {y:>9.6f}, {z:>9.6f})"
        )

    oxygen_indices = [
        index
        for index, site in enumerate(structure)
        if site.species_string == "O"
    ]

    print("\nOxygen site indices")
    print(oxygen_indices)


if __name__ == "__main__":
    main()