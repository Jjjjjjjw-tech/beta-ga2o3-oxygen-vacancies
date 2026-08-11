from pathlib import Path
import sys


# ============================================================
# Add project root to Python path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Import project configuration
# ============================================================

from config import (
    PROJECT_ROOT,
    ANALYSIS_DIR,
    DATA_ROOT,
    PRISTINE_SCF_DIR,
    VACANCY_O1_SCF_DIR,
    VACANCY_O2_SCF_DIR,
    VACANCY_O3_SCF_DIR,
)


# ============================================================
# Print configuration
# ============================================================

print("=" * 60)
print("Project")
print("=" * 60)

print(f"PROJECT_ROOT : {PROJECT_ROOT}")
print(f"ANALYSIS_DIR : {ANALYSIS_DIR}")

print()

print("=" * 60)
print("Raw data")
print("=" * 60)

print(f"DATA_ROOT : {DATA_ROOT}")

print()

print(f"Pristine : {PRISTINE_SCF_DIR}")
print(f"O1       : {VACANCY_O1_SCF_DIR}")
print(f"O2       : {VACANCY_O2_SCF_DIR}")
print(f"O3       : {VACANCY_O3_SCF_DIR}")

print()

print("=" * 60)
print("Files exist")
print("=" * 60)

files_to_check = {
    "Pristine vasprun.xml":
        PRISTINE_SCF_DIR / "vasprun.xml",

    "O1 vasprun.xml":
        VACANCY_O1_SCF_DIR / "vasprun.xml",

    "O2 vasprun.xml":
        VACANCY_O2_SCF_DIR / "vasprun.xml",

    "O3 vasprun.xml":
        VACANCY_O3_SCF_DIR / "vasprun.xml",
}

for name, path in files_to_check.items():
    print(
        f"{name:24s}: "
        f"{path.exists()}"
    )