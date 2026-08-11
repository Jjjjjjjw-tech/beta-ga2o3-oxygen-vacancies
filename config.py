"""
Project configuration

Only modify DATA_ROOT when the raw VASP data are moved
to another SSD or another computer.
"""

from pathlib import Path

# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

STRUCTURES_DIR = PROJECT_ROOT / "structures"
VASP_INPUTS_DIR = PROJECT_ROOT / "vasp_inputs"


# ============================================================
# Analysis folders
# ============================================================

BONDING_DIR = ANALYSIS_DIR / "bonding"
CHARGE_DIR = ANALYSIS_DIR / "charge"
CORRELATION_DIR = ANALYSIS_DIR / "correlation"
ENERGY_DIR = ANALYSIS_DIR / "energy"
FORMATION_ENERGY_DIR = ANALYSIS_DIR / "formation_energy"
LOCAL_STRUCTURE_DIR = ANALYSIS_DIR / "local_structure"
RELAXATION_DIR = ANALYSIS_DIR / "relaxation"
ELECTRONIC_STRUCTURE_DIR = ANALYSIS_DIR / "electronic_structure"


# ============================================================
# Figure folders
# ============================================================

BONDING_FIGURE_DIR = BONDING_DIR / "figures"
CHARGE_FIGURE_DIR = CHARGE_DIR / "figures"
CORRELATION_FIGURE_DIR = CORRELATION_DIR / "figures"

ENERGY_FIGURE_DIR = ENERGY_DIR / "figures"
FORMATION_ENERGY_FIGURE_DIR = FORMATION_ENERGY_DIR / "figures"

LOCAL_STRUCTURE_FIGURE_DIR = LOCAL_STRUCTURE_DIR / "figures"
RELAXATION_FIGURE_DIR = RELAXATION_DIR / "figures"

ELECTRONIC_STRUCTURE_FIGURE_DIR = (
    ELECTRONIC_STRUCTURE_DIR / "figures"
)

DOS_DIR = ELECTRONIC_STRUCTURE_DIR / "dos"
PDOS_DIR = ELECTRONIC_STRUCTURE_DIR / "pdos"
BAND_STRUCTURE_DIR = (
    ELECTRONIC_STRUCTURE_DIR / "band_structure"
)


# ============================================================
# Raw VASP data
# ============================================================

# ------------------------------------------------------------
# ONLY MODIFY THIS PATH WHEN THE DATA LOCATION CHANGES
# ------------------------------------------------------------

DATA_ROOT = Path(
    "/Volumes/Yanchen Li/vasp_results_11.08.2026/data"
)

# ============================================================
# Calculation directories
# ============================================================

PRISTINE_DIR = DATA_ROOT / "pristine"

VACANCY_O1_DIR = DATA_ROOT / "1"

VACANCY_O2_DIR = DATA_ROOT / "2"

VACANCY_O3_DIR = DATA_ROOT / "3"


# ============================================================
# SCF directories
# ============================================================

PRISTINE_SCF_DIR = PRISTINE_DIR / "scf"

VACANCY_O1_SCF_DIR = VACANCY_O1_DIR / "scf"

VACANCY_O2_SCF_DIR = VACANCY_O2_DIR / "scf"

VACANCY_O3_SCF_DIR = VACANCY_O3_DIR / "scf"


# ============================================================
# Helper dictionary
# ============================================================

SCF_DIRS = {
    "pristine": PRISTINE_SCF_DIR,
    "vacancy_01": VACANCY_O1_SCF_DIR,
    "vacancy_02": VACANCY_O2_SCF_DIR,
    "vacancy_03": VACANCY_O3_SCF_DIR,
}


# ============================================================
# Automatically create analysis folders
# ============================================================

OUTPUT_DIRS = [

    BONDING_DIR,
    CHARGE_DIR,
    CORRELATION_DIR,
    ENERGY_DIR,
    FORMATION_ENERGY_DIR,
    LOCAL_STRUCTURE_DIR,
    RELAXATION_DIR,
    ELECTRONIC_STRUCTURE_DIR,

    BONDING_FIGURE_DIR,
    CHARGE_FIGURE_DIR,
    CORRELATION_FIGURE_DIR,
    ENERGY_FIGURE_DIR,
    FORMATION_ENERGY_FIGURE_DIR,
    LOCAL_STRUCTURE_FIGURE_DIR,
    RELAXATION_FIGURE_DIR,
    ELECTRONIC_STRUCTURE_FIGURE_DIR,

    DOS_DIR,
    PDOS_DIR,
    BAND_STRUCTURE_DIR,
]

for folder in OUTPUT_DIRS:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )