from pymatgen.core import Structure

structure = Structure.from_file("structures/beta-Ga2O3_mp-886.cif")

print(structure)
site = structure[0]

print(site)
print(site.specie)
print(site.frac_coords)
print(site.coords)