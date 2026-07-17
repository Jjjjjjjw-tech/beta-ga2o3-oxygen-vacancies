from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar


# 输入文件
vacancy_files = [
    Path("structures/beta-Ga2O3_vacancy_O1.cif"),
    Path("structures/beta-Ga2O3_vacancy_O2.cif"),
    Path("structures/beta-Ga2O3_vacancy_O3.cif"),
]

# 输出目录
output_dir = Path("vasp_inputs")
output_dir.mkdir(exist_ok=True)


for filename in vacancy_files:
    # 读取 CIF 文件
    structure = Structure.from_file(filename)

    # 将 Structure 转换为 Poscar 对象
    poscar = Poscar(structure)

    # beta-Ga2O3_vacancy_O1 -> O1
    vacancy_label = filename.stem.split("_")[-1]

    # 输出文件，例如 vasp_inputs/POSCAR_O1
    output_file = output_dir / f"POSCAR_{vacancy_label}"

    # 写出 POSCAR
    poscar.write_file(output_file)

    print(f"Saved: {output_file}")