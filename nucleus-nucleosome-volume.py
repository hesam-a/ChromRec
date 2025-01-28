import math

nucleus_diameter = 10e-6      # in meters
nucleosome_diameter = 11e-9  # in meters
total_genome_bp = 3e9
bp_per_nuc = 200
vol_of_nuc = (4/3)*math.pi*(nucleosome_diameter/2)**3
total_genome_nucleosomes = total_genome_bp / bp_per_nuc
total_vol_nuc_packed = total_genome_nucleosomes * vol_of_nuc

print(f"\ndiameter of a nucleus: {nucleus_diameter} meters or 10 micrometers\n")
print(f" diameter of a nucleosome: {nucleosome_diameter} meters or 11 nanometers\n")
print(f" Total number of a base pairs in genome: {total_genome_bp}\n")
print(f" Total number of a base pairs per nucleosome + link DNA: {bp_per_nuc}\n")
print(f" Total number of nucleosomes in genome: {total_genome_nucleosomes}\n")
print(f" Volume of nucleosome: {vol_of_nuc:.2e} meters**3\n")
print(f" Total volume of nucleosomes if packed with no space: {total_vol_nuc_packed:.2e} meters**3\n")

volume_nucleosomes_total = 10_453_649_554_820  # Total volume of all nucleosomes in Å³

# Calculate the radius of a sphere with the given volume
# Volume of a sphere: V = (4/3) * pi * R^3
# Solve for R: R = ( (3 * V) / (4 * pi) )^(1/3)
#radius = ((3 * volume_nucleosomes_total) / (4 * math.pi)) ** (1 / 3)
radius = ((3 * total_vol_nuc_packed) / (4 * math.pi)) ** (1 / 3)
print(f" Radius of the volume of all nucleosomes packed: {radius:.2e} in meters\n")

# Convert radius from Å to micrometers (1 Å = 1e-4 micrometers)
radius_micrometers = radius * 1e6
print(f" Radius of the volume of all nucleosomes packed: {radius_micrometers:.2e} in micrometers\n")

# Diameter of the nucleus in micrometers
diameter_micrometers = 2 * radius_micrometers

print(f" diameter of sphere packed by nucleosomes is {diameter_micrometers:.2f} micrometers")
print(f" where the experimental diameter of a nucleus is {nucleus_diameter*1e6} micrometers")
print(f" {diameter_micrometers:.2f} multiplied by 3.7 equals {nucleus_diameter*1e6} which shows there are spaces between nucleosomes\n")
