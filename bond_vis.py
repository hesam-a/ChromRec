import pymol, sys
from pymol import cmd

# Initialize PyMOL
pymol.finish_launching()

# Clear PyMOL
cmd.delete('all')

# Load your structure file
cmd.load('octamer.xyz', 'dna')

# Define the bonds with correct atom IDs (1-based indexing)
dna_bonds = [
    (22, 9),
    (9 , 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (20, 21),
    (21, 23)
    ]

# Hide everything to start with a clean slate
#cmd.hide('everything', 'dna')

# Function to select particles by 1-based index
def select_particle(particle_id):
    return f'dna and index {particle_id}'

# Draw the chain by bonding the specified DNA particles
for start, end in dna_bonds:
    cmd.bond(select_particle(start), select_particle(end))

# Create groups for histones and DNA particles
histone_indices = [1, 2, 3, 4, 5, 6, 7, 8]
dna_indices = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

# Create selections for histones and DNA particles
cmd.select('histones', ' or '.join([select_particle(i) for i in histone_indices]))
cmd.select('dna_particles', ' or '.join([select_particle(i) for i in dna_indices]))

# Set representations
cmd.show('sticks', 'dna_particles')
cmd.color('white', 'dna_particles')
cmd.set('stick_radius', '7', 'dna_particles')


# For histones, use spheres with a specific radius
cmd.show('spheres', 'histones')
cmd.color('red', 'histones')
cmd.set('sphere_scale', 1.0)  # Set base scale for spheres

# Adjust sphere radii for histones and DNA particles
for i in histone_indices:
    cmd.alter(f'dna and index {i}', 'vdw=16.5')
for i in dna_indices:
    cmd.alter(f'dna and index {i}', 'vdw=14.5')

# Apply changes
cmd.rebuild()

# Set a white background
cmd.bg_color('white')

# Center the view on the DNA and histones
cmd.center('dna_particles or histones')

# Refresh the display
cmd.refresh()
