import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

np.set_printoptions(suppress=True, precision=8)

nuc = np.array([
    [131.212189, 126.855835, 137.335938],  #  H3
    [129.095459, 116.979622, 136.993576],  #  H4
    [103.079391, 105.861420, 119.854347],  #  H2A
    [109.069916,  98.158401, 119.412384],  #  H2B
    [107.114456, 133.206940, 110.713539],  #  H3
    [112.637611, 127.008202, 104.648575],  #  H4
    [142.811447, 116.968361, 112.810295],  #  H2A
    [140.466858, 110.083466, 106.688095],  #  H2B
    [ 70.801989, 164.146089, 135.811757],  #  DNA1
    [ 77.539348, 127.626739, 113.566544],  #  DNA2
    [ 94.799382,  93.011429, 101.357632],  #  DNA3
    [126.132181,  76.905611, 118.683504],  #  DNA4
    [146.078879, 101.263044, 146.419319],  #  DNA5
    [135.713069, 137.853976, 152.103405],  #  DNA6
    [108.596185, 154.979404, 128.290182],  #  DNA7
    [ 93.795476, 138.792116,  94.388806],  #  DNA8
    [112.370573, 104.831360,  83.067242],  #  DNA9
    [143.317428,  87.908851, 101.119288],  #  DNA10
    [161.829233, 110.717803, 128.455792],  #  DNA11
    [154.885864, 151.744961, 137.070471],  #  DNA12
    [138.634265, 186.337816, 135.966442]  #  DNA13
#    [ 64.576464, 197.891079, 156.367010],  #  DNA14
#    [121.632948, 222.526506, 134.811482]   #  DNA15
])

dna_indices = list(range(8, 23))

def dir_vec(particle1, particle2):
    return (particle2 - particle1) / np.linalg.norm(particle2 - particle1)


def rotation_matrix(axis, theta):
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                     [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])

def in_plane_rotation_matrix(theta, normal):
    K = np.array([
        [0, -normal[2], normal[1]],
        [normal[2], 0, -normal[0]],
        [-normal[1], normal[0], 0]
    ])
    I = np.identity(3)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R

def rotate_particles(particles, R, origin):
    translated_particles = particles - origin
    rotated_particles = np.dot(translated_particles, R.T)
    rotated_particles += origin
    return rotated_particles

def rotate_particles_fix_three(particles, fixed_indices, theta):
    fixed_points = particles[fixed_indices]
    axis = dir_vec(fixed_points[0], fixed_points[-1])
    R = rotation_matrix(axis, theta)
    rotated_particles = particles.copy()
    for i, particle in enumerate(particles):
        if i not in fixed_indices:
            translated_particle = particle - fixed_points[0]
            rotated_particle = np.dot(translated_particle, R.T)
            rotated_particle += fixed_points[0]
            rotated_particles[i] = rotated_particle
    return rotated_particles

def add_linker_dna(particles, num_linkers):
    exit_dna = particles[-1]  # DNA13
    direction = dir_vec(particles[-2], exit_dna)  # Direction from DNA12 to DNA13
    new_particles = particles.copy()
    for i in range(1, num_linkers + 1):
        new_particle = exit_dna + i * 40 * direction  # 40 Å per DNA particles
        new_particles = np.vstack([new_particles, new_particle])
    return new_particles


def rand_gen(mu, stddev, items):
    # Create a large sample of Gaussian-distributed values
    gaussian_samples = np.random.normal(mu, stddev, 1000000)
    # Clip the values to the range of the list indices
    gaussian_samples = np.clip(gaussian_samples, 0, len(items) - 1)
    # Round to nearest integer and ensure all indices are within the valid range
    indices = np.round(gaussian_samples).astype(int)
    # Adjust the distribution by normalizing the PMF
    unique, counts = np.unique(indices, return_counts=True)
    pmf = counts / sum(counts)
    # Create a dictionary for the probability of each item
    probabilities = {item: pmf[i] for i, item in enumerate(unique)}
    # To ensure the probability distribution matches the length of items
    item_probabilities = [probabilities.get(item, 0) for item in items]
    # Normalize the probabilities to ensure they sum to 1
    item_probabilities /= np.sum(item_probabilities)

    return item_probabilities

def generate_globular_chain(num_points, step_size, radial_distance, max_attempts=1200, backtrack_steps=5, center_bias=0.03):
    points = [np.array([0.0, 0.0, 0.0])]

    current_particles = nuc.copy()

    angle_d = np.deg2rad(10)
    angle_d2 = np.deg2rad(20)

    g = open("chromatin_testtt.xyz", 'w')
    f = open("fiber_test.test", "w")

    # Initialize starting indices
    atom_idx_start  = 1
    bond_idx_start  = 1
    angle_idx_start = 1

    # Initialize lists for storing atoms, bonds, and angles
    atoms = []
    bonds = []
    angles = []
    generated_nuc = []    
    natoms = 0

    # Parameters of rand_gen
    items = [0, 1, 2, 3, 4, 5]
    mu = 2.5  # Gaussian mean for rand_gen
    sigma = 1.35  # std dev for rand_gen
    
    pmf = rand_gen(mu, sigma, items)


    i = 1
    while i < num_points + 1:
        found_valid = False
        attempts = 0
        while not found_valid and attempts < max_attempts:
            attempts += 1
            # Random step in 3D
            step = np.random.normal(size=3)
            step /= np.linalg.norm(step) 
            step *= step_size

            new_point = points[-1] + step

            # Bias towards center of mass
            center_of_mass = np.mean(points, axis=0)
            direction_to_center = center_of_mass - new_point
            new_point += center_bias * direction_to_center
            #print(f" difference after bias:   {np.linalg.norm(new_point - (points[-1] + step)):.2f}")

            if all(np.linalg.norm(new_point - p) >= radial_distance for p in points):
                found_valid = True
                points.append(new_point)

        if found_valid:
            # Randomly select angles within the range [angle - angle_d, angle + angle_d]
            theta1 = (np.pi + random.uniform(-angle_d, angle_d)) * random.choice([-1, 1])
            theta2 = (np.deg2rad(30) + random.uniform(-angle_d, angle_d)) * random.choice([-1, 1])
            theta3 = (np.deg2rad(45) + random.uniform(-angle_d2, angle_d2)) * random.choice([-1, 1])

            # Define the axes for rotation
            axis1 = dir_vec(current_particles[10], current_particles[12])
            axis2 = dir_vec(current_particles[15], current_particles[18])

            # First rotation
            R1 = rotation_matrix(axis1, theta1)
            rotated_particles = rotate_particles(current_particles, R1, current_particles[12])

            # Second rotation (in-plane)
            fixed_point = rotated_particles[12]
            v1 = rotated_particles[10] - fixed_point
            v2 = rotated_particles[14] - fixed_point
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            R2 = in_plane_rotation_matrix(theta2, normal)
            rotated_nuc = rotate_particles(rotated_particles, R2, fixed_point)

            # Third rotation with fixed particles
            fixed_indices = [19, 20]
            rotated_plane = rotate_particles_fix_three(rotated_nuc, fixed_indices, theta3)
            generated_nuc = rotated_plane            

            # Add random linker DNA particles
            num_linkers = np.random.choice(items, p=pmf)
            print(f" num_linkers: {num_linkers}")
            rotated_plane = add_linker_dna(rotated_plane, num_linkers)
            print(f" len of rotated_plane: {len(rotated_plane)}")

            # Translate the rotated particles to the new generated point
            rotated_plane_center_of_mass = np.mean(rotated_plane, axis=0)
            translation_vector = points[-1] - rotated_plane_center_of_mass
            translated_particles = rotated_plane + translation_vector

            natoms += len(translated_particles) 

            # Append atoms
            for j in range(len(translated_particles)):
                atom_type = 5 if j >= 8 else ((j % 4) + 1)
                atoms.append(f"   {atom_idx_start+j:<8}{i:<6}{atom_type:<5}{translated_particles[j][0]:<12.5f}{translated_particles[j][1]:<12.5f}{translated_particles[j][2]:<12.5f}{atom_comment[j]}\n")

            # Update atom index start for next nucleosome
            atom_idx_start += len(translated_particles)

            # Append bonds
            for bond_idx, bond in enumerate(original_bonds):
                bonds.append(f"   {bond_idx_start + bond_idx:<6}{int(bond[0]):<6}{int(bond[1])+natoms-len(translated_particles):<6}{int(bond[2])+natoms-len(translated_particles):<6}{bond[3]:<3}\n")                
                #bonds.append(f"   {bond_idx_start + bond_idx:<8}{int(bond[0]):<6}{int(bond[1])+((i-1)*23):<6}{int(bond[2])+((i-1)*23):<6}{bond[3]:<3}\n")

            # Update bond index start for next nucleosome
            bond_idx_start += len(original_bonds)

            # Append linker DNA bonds
            for idx in range(num_linkers):
                bonds.append(f"   {bond_idx_start:<6}{int(linker_DNA_bonds[idx][0]):<6}{int(linker_DNA_bonds[idx][1])+natoms-len(translated_particles):<6}{int(linker_DNA_bonds[idx][2])+natoms-len(translated_particles):<6}{linker_DNA_bonds[idx][3]:<3} - Linker DNAs\n")
                bond_idx_start += 1

            if i < num_points:
                # Connect DNA14 of the generated nucleosome to DNA15 of the previous one
                bonds.append(f"   {bond_idx_start:<6}{str(14):<6}{natoms:<6}{natoms+9:<3}   #  Connecting DNAs\n")
                #bonds.append(f"   {bond_idx_start:<8}{str(14):<6}{atom_idx_start-len(translated_particles)+22:<6}{atom_idx_start+len(translated_particles)-2:<3}\n")
                bond_idx_start += 1
                

            # Append angles
            for angle_idx, angle in enumerate(original_angles):
                angles.append(f"   {angle_idx_start + angle_idx:<6}{int(angle[0]):<6}{int(angle[1])+natoms-len(translated_particles):<6}{int(angle[2])+natoms-len(translated_particles):<6}{int(angle[3])+natoms-len(translated_particles):<6}{angle[4]:<3}\n")

            # Update angle index start for next nucleosome
            angle_idx_start += len(original_angles)

            # Append linker DNA angles
            for idx in range(num_linkers):
                angles.append(f"   {angle_idx_start:<6}{int(linker_DNA_angles[idx][0]):<6}{int(linker_DNA_angles[idx][1])+natoms-len(translated_particles):<6}{int(linker_DNA_angles[idx][2])+natoms-len(translated_particles):<6}{int(linker_DNA_angles[idx][3])+natoms-len(translated_particles):<6}{linker_DNA_angles[idx][4]:<3} - Linker DNAs\n")
                angle_idx_start += 1

            if i < num_points-1:
                # Add connecting angles
             #angles.append(f"   {angle_idx_start:<6}{str(1):<6}{atom_idx_start-len(translated_particles)+20:<6}{atom_idx_start-len(translated_particles)+22:<6}{atom_idx_start+len(translated_particles)-2:<3}\n")
                angles.append(f"   {angle_idx_start:<6}{str(1):<6}{natoms-1:<6}{natoms:<6}{natoms+9:<6}#  Connecting DNAs\n")
                angle_idx_start += 1
                #angles.append(f"   {angle_idx_start:<6}{str(1):<6}{atom_idx_start-len(translated_particles)+22:<6}{atom_idx_start+len(translated_particles)-2:<6}{atom_idx_start+8:<3}\n")
                angles.append(f"   {angle_idx_start:<6}{str(1):<6}{natoms:<6}{natoms+9:<6}{natoms+10:<6}#  Connecting DNAs\n")
                angle_idx_start += 1


            #for j in range(len(translated_particles)):
            #    g.write(f"   C   {translated_particles[j][0]:<15.5f}{translated_particles[j][1]:<15.5f}{translated_particles[j][2]:<15.5f}\n")

            #current_particles = translated_particles
            current_particles = generated_nuc


            print(f"Generated point {i}/{num_points}")
            i += 1

        else:
            # Backtrack if too many attempts have been made
            print(f"Backtracking from point {i} after {attempts} attempts")
            for _ in range(min(backtrack_steps, len(points)-1)):
                points.pop()

            # Backtrack atoms
            for _ in range(min((backtrack_steps) * len(nuc), len(atoms))):
                atoms.pop()

            # Backtrack bonds
            bonds_per_nucleosome = len(original_bonds) + 1 
            for _ in range(min((backtrack_steps) * bonds_per_nucleosome, len(bonds))):
                bonds.pop()

            # Backtrack angles
            angles_per_nucleosome = len(original_angles) + 2 
            for _ in range(min(backtrack_steps * angles_per_nucleosome, len(angles))):
                angles.pop()

            i = max(1, i - backtrack_steps)
            atom_idx_start  = max(1, atom_idx_start - backtrack_steps * len(nuc))
            bond_idx_start  = max(1, bond_idx_start - backtrack_steps * (len(original_bonds)+1))
            angle_idx_start = max(1, angle_idx_start - backtrack_steps * (len(original_angles)+2))

    num_atoms = 0
    for i, nucs in enumerate(atoms):
        num_atoms += len(nucs)

    data = np.array([list(map(float, line.split()[3:6])) for line in atoms])

    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
    z_min, z_max = np.min(data[:, 2]), np.max(data[:, 2])
    padding = 200

    g.write(f"{str(natoms)}\n\n")

    f.write("# LAMMPS data file\n\n")

    f.write(f"   {str(natoms)}   atoms\n")
    f.write(f"   {str(len(bonds))}   bonds\n")
    f.write(f"   {str(len(angles))}   angles\n\n")

    f.write("   5    atom types \n")
    f.write("   40   bond types \n")
    f.write("   1    angle types\n\n")
    #f.write("   2    angle types\n\n")

    f.write(f"   {str(round(x_min - padding, 2))}  {str(round(x_max + padding, 2))} xlo xhi\n")
    f.write(f"   {str(round(y_min - padding, 2))}  {str(round(y_max + padding, 2))} ylo yhi\n")
    f.write(f"   {str(round(z_min - padding, 2))}  {str(round(z_max + padding, 2))} zlo zhi\n\n")

    f.write("Masses\n\n")
    f.write("   1    15273\n")
    f.write("   2    11390\n")
    f.write("   3    13789\n")
    f.write("   4    13990\n")
    f.write("   5     9267\n")

    f.write("\nAtoms\n\n")
    for atom in atoms:
        g.write(f"{atom.split()[2]:<8}{float(atom.split()[3]):<12.5f}{float(atom.split()[4]):<12.5f}{float(atom.split()[5]):<12.5f} {atom.split()[6]}\n") 
        f.write(atom)

    f.write("\nBonds\n\n")
    for bond in bonds:
        f.write(bond)

    f.write("\nAngles\n\n")
    for angle in angles:
        f.write(angle)

    return np.array(points)



# Define the bond and angle types from the first nucleosome
original_bonds = np.array([
    [1 , 1,  2,  "#  A - B"],
    [1 , 5,  6,  "#  E - F"],
    [2 , 3,  4,  "#  C - D"],
    [2 , 7,  8,  "#  G - H"],
    [3 , 1,  5,  "#  A - E"],
    [4 , 2,  6,  "#  B - F"],
    [5 , 3,  7,  "#  C - G"],
    [6 , 4,  8,  "#  D - H"],
    [7 , 2,  4,  "#  B - D"],
    [7 , 2,  8,  "#  B - H"],
    [7 , 6,  4,  "#  F - D"],
    [7 , 6,  8,  "#  F - H"],
    [8 , 1,  7,  "#  A - G"],
    [8 , 5,  3,  "#  E - C"],
    [9 , 5,  7,  "#  E - G"],
    [9 , 1,  3,  "#  A - C"],
    [10, 1,  8,  "#  H3-H2B   1-8"],
    [10, 5,  4,  "#  H3-H2B   5-4"],
    [11, 5,  8,  "#  H3-H2B   5-8"],
    [11, 1,  4,  "#  H3-H2B   1-4"],
    [12, 2,  7,  "#  H4-H2A   2-7"],
    [12, 6,  3,  "#  H4-H2A   6-3"],
    [13, 2,  3,  "#  H4-H2A   2-3"],
    [13, 6,  7,  "#  H4-H2A   6-7"],
    [14, 9,  10, "#  DNA"],
    [14, 10, 11, "#  DNA"],
    [14, 11, 12, "#  DNA"],
    [14, 12, 13, "#  DNA"],
    [14, 13, 14, "#  DNA"],
    [14, 14, 15, "#  DNA"],
    [14, 15, 16, "#  DNA"],
    [14, 16, 17, "#  DNA"],
    [14, 17, 18, "#  DNA"],
    [14, 18, 19, "#  DNA"],
    [14, 19, 20, "#  DNA"],
    [14, 20, 21, "#  DNA"],
    [15, 10, 5,  "#  DNA - H3"],
    [16, 10, 3,  "#  DNA - H2A"],
    [17, 11, 3,  "#  DNA - H2A"],
    [18, 11, 6,  "#  DNA - H2B"],
    [19, 12, 4,  "#  DNA - H2B"],
    [20, 12, 8,  "#  DNA - H2B"],
    [21, 13, 2,  "#  DNA - H4"],
    [22, 13, 7,  "#  DNA - H2A"],
    [23, 14, 1,  "#  DNA - H3"],
    [24, 14, 7,  "#  DNA - H4"],
    [25, 15, 1,  "#  DNA - H3"],
    [26, 15, 5,  "#  DNA - H3"],
    [27, 16, 3,  "#  DNA - H3"],
    [28, 16, 5,  "#  DNA - H4"],
    [29, 17, 6,  "#  DNA - H4"],
    [30, 17, 4,  "#  DNA - H2B"],
    [31, 18, 8,  "#  DNA - H2B"],
    [32, 18, 4,  "#  DNA - H3"],
    [33, 19, 1,  "#  DNA - H3"],
    [34, 19, 7,  "#  DNA - H2A"],
    [35, 20, 7,  "#  DNA - H2A"],
    [36, 20, 1,  "#  DNA - H3"],
    [37, 9 , 15, "#  DNA"],
    [38, 15, 21, "#  DNA"],
    [39, 10, 15, "#  DNA"],
    [40, 15, 20, "#  DNA"]])

linker_DNA_bonds = np.array([
    [14, 21, 22, "#  DNA13 - DNA14"],
    [14, 22, 23, "#  DNA14 - DNA15"],
    [14, 23, 24, "#  DNA15 - DNA16"],
    [14, 24, 25, "#  DNA16 - DNA17"],
    [14, 25, 26, "#  DNA17 - DNA18"],
    [14, 26, 27, "#  DNA18 - DNA19"]])

original_angles = np.array([
#    [1, 22, 9 , 10], 
    [1,  9, 10, 11, "#  DNA1 - DNA2 - DNA3"],
    [1, 10, 11, 12, "#  DNA2 - DNA3 - DNA4"],
    [1, 11, 12, 13, "#  DNA3 - DNA4 - DNA5"],
    [1, 12, 13, 14, "#  DNA4 - DNA5 - DNA6"],
    [1, 13, 14, 15, "#  DNA5 - DNA6 - DNA7"],
    [1, 14, 15, 16, "#  DNA6 - DNA7 - DNA8"],
    [1, 15, 16, 17, "#  DNA7 - DNA8 - DNA9"],
    [1, 16, 17, 18, "#  DNA8 - DNA9 - DNA10"],
    [1, 17, 18, 19, "#  DNA9 - DNA10 - DNA11"],
    [1, 18, 19, 20, "#  DNA10 - DNA11 - DNA12"],
    [1, 19, 20, 21, "#  DNA11 - DNA12 - DNA13"]])
#    [2, 9 , 15, 21]])
#   [3, 10, 15, 20]]

linker_DNA_angles = np.array([
    [1, 20, 21, 22, "#  DNA12 - DNA13 - DNA14"],
    [1, 21, 22, 23, "#  DNA13 - DNA14 - DNA15"],
    [1, 22, 23, 24, "#  DNA14 - DNA15 - DNA16"],
    [1, 23, 24, 25, "#  DNA15 - DNA16 - DNA17"],
    [1, 24, 25, 26, "#  DNA16 - DNA17 - DNA18"],
    [1, 25, 26, 27, "#  DNA17 - DNA18 - DNA19"],
    ])

atom_comment = [" #  H3  ",  " #  H4  ",  " #  H2A ",  " #  H2B ",  " #  H3  ",  " #  H4  ",  " #  H2A ",
                " #  H2B ",  " #  DNA1",  " #  DNA2",  " #  DNA3",  " #  DNA4",  " #  DNA5",  " #  DNA6",
                " #  DNA7",  " #  DNA8",  " #  DNA9",  " #  DNA10", " #  DNA11", " #  DNA12", " #  DNA13",
                " #  DNA14", " #  DNA15", " #  DNA16", " #  DNA17", " #  DNA18", " #  DNA19", " #  DNA20"]


# Parameters
num_points = 100 
step_size = 160.0
radial_distance = 140


# Generate chain
chain = generate_globular_chain(num_points, step_size, radial_distance)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], marker='o', linestyle='-', color='b', markerfacecolor='r')
#ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], marker='o', linestyle='-', color='red', markerfacecolor='yellow', markeredgecolor='black')
#ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], marker='o', linestyle='-', color='darkblue', markerfacecolor='white', markeredgecolor='orange')
#ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], marker='o', linestyle='-', color='green', markerfacecolor='pink', markeredgecolor='purple')
#ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], marker='o', linestyle='-', color='orange', markerfacecolor='lightblue', markeredgecolor='navy')
#ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], marker='o', linestyle='-', color='purple', markerfacecolor='lime', markeredgecolor='red')
plt.show()
