from pathlib import Path

from pymatgen.core import Structure


def main() -> None:
    input_path = Path("structures") / "beta-Ga2O3_mp-886.cif"
    structure = Structure.from_file(input_path)

    representative_oxygen_indices = [4, 6, 8]

    for oxygen_type, oxygen_index in enumerate(
        representative_oxygen_indices,
        start=1,
    ):
        vacancy_structure = structure.copy()

        selected_site = vacancy_structure[oxygen_index]

        if selected_site.species_string != "O":
            raise ValueError(
                f"Site {oxygen_index} is not oxygen: "
                f"{selected_site.species_string}"
            )

        vacancy_structure.remove_sites([oxygen_index])

        output_path = (
            Path("structures")
            / f"beta-Ga2O3_vacancy_O{oxygen_type}.cif"
        )

        vacancy_structure.to(
            filename=str(output_path),
            fmt="cif",
        )

        print(
            f"O{oxygen_type}: removed oxygen index "
            f"{oxygen_index}, saved to {output_path}"
        )
        print(
            f"Atoms: {len(structure)} -> "
            f"{len(vacancy_structure)}"
        )


if __name__ == "__main__":
    main()