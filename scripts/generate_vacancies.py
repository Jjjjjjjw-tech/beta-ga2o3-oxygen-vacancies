from pathlib import Path

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def main() -> None:
    # 原始无缺陷结构
    input_path = Path("structures") / "beta-Ga2O3_mp-886.cif"

    # 输出目录
    output_dir = Path("structures") / "supercell"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取原始晶体结构
    primitive_structure = Structure.from_file(input_path)

    print(f"Original structure: {primitive_structure.composition}")
    print(f"Original number of atoms: {len(primitive_structure)}")

    # --------------------------------------------------
    # 1. 创建 2 × 2 × 2 超胞
    # Ga4O6 → Ga32O48
    # --------------------------------------------------
    supercell = primitive_structure.copy()
    supercell.make_supercell([2, 2, 2])

    print(f"Supercell composition: {supercell.composition}")
    print(f"Supercell number of atoms: {len(supercell)}")

    # 保存完整、无缺陷的超胞
    pristine_path = output_dir / "beta-Ga2O3_pristine_2x2x2.cif"
    supercell.to(filename=str(pristine_path), fmt="cif")

    print(f"Saved pristine supercell to: {pristine_path}")

    # --------------------------------------------------
    # 2. 在超胞中重新寻找不等价氧位点
    # --------------------------------------------------
    symmetry_analyzer = SpacegroupAnalyzer(
        supercell,
        symprec=0.01,
        angle_tolerance=5,
    )

    symmetrized_structure = symmetry_analyzer.get_symmetrized_structure()

    representative_oxygen_indices: list[int] = []

    for equivalent_indices in symmetrized_structure.equivalent_indices:
        representative_index = equivalent_indices[0]
        representative_site = supercell[representative_index]

        if representative_site.species_string == "O":
            representative_oxygen_indices.append(representative_index)

    print(
        "Representative oxygen indices in supercell:",
        representative_oxygen_indices,
    )

    if len(representative_oxygen_indices) != 3:
        raise ValueError(
            "Expected 3 inequivalent oxygen sites, but found "
            f"{len(representative_oxygen_indices)}. "
            "Please check the symmetry tolerance or input structure."
        )

    # --------------------------------------------------
    # 3. 分别删除三类氧位点中的一个氧
    # --------------------------------------------------
    for oxygen_type, oxygen_index in enumerate(
        representative_oxygen_indices,
        start=1,
    ):
        vacancy_structure = supercell.copy()

        selected_site = vacancy_structure[oxygen_index]

        if selected_site.species_string != "O":
            raise ValueError(
                f"Site {oxygen_index} is not oxygen: "
                f"{selected_site.species_string}"
            )

        print(
            f"O{oxygen_type}: removing oxygen index {oxygen_index}, "
            f"fractional coordinates = {selected_site.frac_coords}"
        )

        vacancy_structure.remove_sites([oxygen_index])

        output_path = (
            output_dir
            / f"beta-Ga2O3_vacancy_O{oxygen_type}_2x2x2.cif"
        )

        vacancy_structure.to(
            filename=str(output_path),
            fmt="cif",
        )

        print(
            f"Saved O{oxygen_type} vacancy structure: "
            f"{vacancy_structure.composition}"
        )
        print(f"Saved to: {output_path}")

    print("\nVacancy generation completed.")


if __name__ == "__main__":
    main()