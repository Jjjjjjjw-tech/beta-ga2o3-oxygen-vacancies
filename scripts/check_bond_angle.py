#!/usr/bin/env python3
"""Check selected O-Ga-O angles in the relaxed pristine structure."""

from pathlib import Path

from pymatgen.core import Structure


# ======================================
# Configuration
# ======================================

RESULT_DATE = "2026-07-25_results"

GA_ATOM_NUMBER = 29

OXYGEN_ATOM_NUMBERS = (
    37,
    61,
    71,
)


# ======================================
# Build file path
# ======================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRISTINE_CONTCAR_PATH = (
    PROJECT_ROOT
    / "beta-Ga2O3_oxygen_vacancy_VASP_inputs"
    / "received_results"
    / RESULT_DATE
    / "pristine"
    / "relax"
    / "CONTCAR"
)


# ======================================
# Load structure
# ======================================

if not PRISTINE_CONTCAR_PATH.is_file():
    raise FileNotFoundError(
        f"Cannot find pristine CONTCAR:\n"
        f"{PRISTINE_CONTCAR_PATH}"
    )

pristine = Structure.from_file(
    PRISTINE_CONTCAR_PATH
)


# ======================================
# Convert atom numbers to Python indices
# ======================================

ga_index = GA_ATOM_NUMBER - 1

oxygen_indices = [
    atom_number - 1
    for atom_number in OXYGEN_ATOM_NUMBERS
]


# ======================================
# Check atom identities
# ======================================

ga_site = pristine[ga_index]

if ga_site.species_string != "Ga":
    raise ValueError(
        f"Atom {GA_ATOM_NUMBER} is "
        f"{ga_site.species_string}, not Ga."
    )

for atom_number, oxygen_index in zip(
    OXYGEN_ATOM_NUMBERS,
    oxygen_indices,
):
    oxygen_site = pristine[oxygen_index]

    if oxygen_site.species_string != "O":
        raise ValueError(
            f"Atom {atom_number} is "
            f"{oxygen_site.species_string}, not O."
        )


# ======================================
# Print coordinates
# ======================================

print("=" * 70)
print("Selected atomic sites")
print("=" * 70)

print(
    f"Ga{GA_ATOM_NUMBER} "
    f"(Python index {ga_index})"
)

print(
    f"  Fractional coordinates: "
    f"{ga_site.frac_coords}"
)

print(
    f"  Cartesian coordinates : "
    f"{ga_site.coords}"
)

for atom_number, oxygen_index in zip(
    OXYGEN_ATOM_NUMBERS,
    oxygen_indices,
):
    oxygen_site = pristine[oxygen_index]

    print()
    print(
        f"O{atom_number} "
        f"(Python index {oxygen_index})"
    )

    print(
        f"  Fractional coordinates: "
        f"{oxygen_site.frac_coords}"
    )

    print(
        f"  Cartesian coordinates : "
        f"{oxygen_site.coords}"
    )

    distance = pristine.get_distance(
        ga_index,
        oxygen_index,
    )

    print(
        f"  Ga-O distance         : "
        f"{distance:.4f} Å"
    )


# ======================================
# Calculate O-Ga-O angles
# ======================================

print()
print("=" * 70)
print("O-Ga-O angles")
print("=" * 70)

for first_position in range(
    len(oxygen_indices)
):
    for second_position in range(
        first_position + 1,
        len(oxygen_indices),
    ):
        oxygen_1_number = (
            OXYGEN_ATOM_NUMBERS[
                first_position
            ]
        )

        oxygen_2_number = (
            OXYGEN_ATOM_NUMBERS[
                second_position
            ]
        )

        oxygen_1_index = (
            oxygen_indices[
                first_position
            ]
        )

        oxygen_2_index = (
            oxygen_indices[
                second_position
            ]
        )

        angle = pristine.get_angle(
            oxygen_1_index,
            ga_index,
            oxygen_2_index,
        )

        print(
            f"O{oxygen_1_number}"
            f"-Ga{GA_ATOM_NUMBER}"
            f"-O{oxygen_2_number}: "
            f"{angle:.4f}°"
        )