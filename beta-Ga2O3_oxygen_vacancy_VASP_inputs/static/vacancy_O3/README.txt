Calculation: beta-Ga2O3 vacancy_O3
Charge state: Neutral (q = 0)

========================================
Pseudopotentials
========================================

Please use the standard VASP PAW-PBE pseudopotentials:

- Ga_d
- O

The POTCAR order must match the POSCAR:

Ga
O

========================================
Step 1: Geometry Relaxation
========================================

Run the calculation in:

relax/vacancy_O3

========================================
Step 2: Static Calculation
========================================

After the relaxation is finished:

1. Copy:
   relax/vacancy_O3/CONTCAR

2. Paste it into:
   static/vacancy_O3/

3. Rename:
   CONTCAR -> POSCAR

4. Run the static calculation.

========================================
Please return the following files
========================================

Relaxation:

- INCAR
- POSCAR
- CONTCAR
- KPOINTS
- OUTCAR
- OSZICAR
- vasprun.xml

Static:

- INCAR
- POSCAR
- KPOINTS
- OUTCAR
- OSZICAR
- vasprun.xml

========================================
Please also report
========================================

- Final total energy (TOTEN)
- VASP version
- POTCAR TITEL entries
- Whether both electronic and ionic convergence were achieved

-If any calculation fails or does not converge,
-please also return all generated output files.

Thank you very much!
