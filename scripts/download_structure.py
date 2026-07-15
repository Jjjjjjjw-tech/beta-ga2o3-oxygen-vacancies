import os
from pathlib import Path

from mp_api.client import MPRester


def main():
    api_key = os.getenv("MP_API_KEY")

    if not api_key:
        raise RuntimeError("MP_API_KEY is not set.")

    output_path = Path("structures") / "beta-Ga2O3_mp-886.cif"

    with MPRester(api_key) as mpr:
        structure = mpr.get_structure_by_material_id("mp-886")

    structure.to(filename=str(output_path), fmt="cif")

    print(f"Saved structure to: {output_path}")
    print(f"Formula: {structure.composition.reduced_formula}")
    print(f"Atoms: {len(structure)}")


if __name__ == "__main__":
    main()