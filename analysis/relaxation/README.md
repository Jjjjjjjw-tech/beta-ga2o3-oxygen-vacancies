# Structural Relaxation Analysis

## Objective

This module analyzes the structural relaxation around neutral oxygen
vacancies in β-Ga2O3 after geometry optimization.

The purpose is to quantify lattice distortion and investigate how local
atomic relaxation influences vacancy stability.

---

## Input

Relaxed VASP calculations:

- pristine
- vacancy_O1
- vacancy_O2
- vacancy_O3

Files used:

- POSCAR
- CONTCAR

---

## Methods

Current analyses include:

- Atomic displacement analysis
- Local relaxation around vacancy sites
- Nearest-neighbor identification
- Radial displacement analysis

All atomic displacements are calculated under periodic boundary
conditions using pymatgen.

Global rigid translation is removed before analysis.

---

## Outputs

### Tables

- atomic_displacements.csv
- relaxation_summary.csv
- local_relaxation_details.csv
- vacancy_site_summary.csv

### Figures

(To be added)

---

## Current Results

### Vacancy O1

- Maximum displacement: 0.227 Å
- Two neighboring Ga atoms relax inward symmetrically.
- Local relaxation appears relatively distributed.

### Vacancy O2

- Maximum displacement: 0.361 Å
- One neighboring Ga atom dominates the relaxation.
- Relaxation is highly localized.

### Vacancy O3

- Maximum displacement: 0.366 Å
- Similar behavior to O2.

---

## Current Interpretation

Current results suggest different local relaxation mechanisms among the
three oxygen vacancies.

However, the relationship between structural relaxation and vacancy
stability has **not yet been fully established**.

Further analyses are required:

- Ga–O bond length changes
- Coordination-number analysis
- Electronic structure (DOS)
- Charge-density analysis

---

## Status

✔ Atomic displacement

✔ Local relaxation

✔  Bond-length analysis

⬜ Coordination analysis

⬜ DOS analysis

⬜ Charge-density analysis