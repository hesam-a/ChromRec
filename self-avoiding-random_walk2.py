import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_globular_chain(num_points, step_size, max_attempts=1000, backtrack_steps=5, center_bias=0.05):
    points = [np.array([0.0, 0.0, 0.0])]  # Starting point

    current_particles = nuc.copy()

    angle_d = np.deg2rad(10)
    angle_d2 = np.deg2rad(20)

    g = open("chromatin_testtt.xyz", 'w')
    f = open("fiber_dataaa.data", "w")
    g.write(f"{str(num_points*23)} \n\n")

    # Initialize starting indices
    atom_idx_start  = 1
    bond_idx_start  = 1
    angle_idx_start = 1

    # Initialize lists for storing atoms, bonds, and angles
    atoms = []
    bonds = []
    angles = []
    

    i = 1
    while i < num_points:
        found_valid = False
        attempts = 0
        while not found_valid and attempts < max_attempts:
            attempts += 1
            # Random step in 3D
            step = np.random.normal(size=3)
            step /= np.linalg.norm(step)  # Normalize to unit length
            step *= step_size

            new_point = points[-1] + step

            # Bias towards center of mass
            center_of_mass = np.mean(points, axis=0)
            direction_to_center = center_of_mass - new_point
            new_point += center_bias * direction_to_center

            if all(np.linalg.norm(new_point - p) >= 230 for p in points):
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
            fixed_indices = [19, 20, 22]
            rotated_plane = rotate_particles_fix_three(rotated_nuc, fixed_indices, theta3)

            # Translate the rotated particles to the new generated point
            rotated_plane_center_of_mass = np.mean(rotated_plane, axis=0)
            translation_vector = points[-1] - rotated_plane_center_of_mass
            translated_particles = rotated_plane + translation_vector

            # Append atoms
            for j in range(len(current_particles)):
                atom_type = 5 if j >= 8 else ((j % 4) + 1)
                atoms.append(f"   {atom_idx_start+j:<8}{i:<6}{atom_type:<5}{current_particles[j][0]:<12.5f}{current_particles[j][1]:<12.5f}{current_particles[j][2]:<12.5f}{atom_comment[j]}\n")

            # Update atom index start for next nucleosome
            atom_idx_start += len(current_particles)

            # Append bonds
            for bond_idx, bond in enumerate(original_bonds):
                bonds.append(f"   {bond_idx_start + bond_idx:<8}{int(bond[0]):<6}{int(bond[1])+((i-1)*23):<6}{int(bond[2])+((i-1)*23):<6}{bond[3]:<3}\n")

            # Update bond index start for next nucleosome
            bond_idx_start += len(original_bonds)

            if i < num_points:
                # Connect DNA14 of the generated nucleosome to DNA15 of the previous one
                bonds.append(f"   {bond_idx_start:<8}{str(14):<6}{atom_idx_start-len(current_particles)+22:<6}{atom_idx_start+len(current_particles)-2:<3}\n")
                bond_idx_start += 1

            # Append angles
            for angle_idx, angle in enumerate(original_angles):
                angles.append(f"   {angle_idx_start + angle_idx:<8}{int(angle[0]):<6}{int(angle[1])+((i-1)*23):<6}{int(angle[2])+((i-1)*23):<6}{int(angle[3])+((i-1)*23):<3}\n")

            # Update angle index start for next nucleosome
            angle_idx_start += len(original_angles)

            if i < num_points:
                # Add connecting angles
                angles.append(f"   {angle_idx_start:<8}{str(1):<6}{atom_idx_start-len(current_particles)+20:<6}{atom_idx_start-len(current_particles)+22:<6}{atom_idx_start+len(current_particles)-2:<3}\n")
                angle_idx_start += 1
                angles.append(f"   {angle_idx_start:<8}{str(1):<6}{atom_idx_start-len(current_particles)+22:<6}{atom_idx_start+len(current_particles)-2:<6}{atom_idx_start+8:<3}\n")
                angle_idx_start += 1

            for j in range(len(current_particles)):
                g.write(f"   C   {current_particles[j][0]:<15.5f}{current_particles[j][1]:<15.5f}{current_particles[j][2]:<15.5f}\n")

            current_particles = translated_particles
            
            
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
            bonds_per_nucleosome = len(original_bonds) + 1  # Include one connecting bond per nucleosome
            for _ in range(min((backtrack_steps) * bonds_per_nucleosome, len(bonds))):
                bonds.pop()

            # Backtrack angles
            angles_per_nucleosome = len(original_angles) + 2  # Include two connecting angles per nucleosome
            for _ in range(min(backtrack_steps * angles_per_nucleosome, len(angles))):
                angles.pop()

            i = max(1, i - backtrack_steps)

    data = np.array([list(map(float, line.split()[3:6])) for line in atoms])

    f.write("# LAMMPS data file\n\n")

    f.write(f"   {str(num_points*23)}   atoms\n")
    f.write(f"   {str(len(bonds))}   bonds\n")
    f.write(f"   {str(len(angles))}   angles\n\n")

    f.write("   5    atom types \n")
    f.write("   40   bond types \n")
    f.write("   1    angle types\n\n")
    #f.write("   2    angle types\n\n")

    f.write(f"   {str(round(min(data.flatten())-20,2))}  {str(round(max(data.flatten())+20,2))} xlo xhi\n")
    f.write(f"   {str(round(min(data.flatten())-20,2))}  {str(round(max(data.flatten())+20,2))} ylo yhi\n")
    f.write(f"   {str(round(min(data.flatten())-20,2))}  {str(round(max(data.flatten())+20,2))} zlo zhi\n\n")

    f.write("Masses\n\n")
    f.write("   1    15273\n")
    f.write("   2    11390\n")
    f.write("   3    13789\n")
    f.write("   4    13990\n")
    f.write("   5     9267\n")

    f.write("\nAtoms\n\n")
    for atom in atoms:
        f.write(atom)

    f.write("\nBonds\n\n")
    for bond in bonds:
        f.write(bond)

    f.write("\nAngles\n\n")
    for angle in angles:
        f.write(angle)

    return np.array(points)

# Parameters
num_points = 1000
step_size = 280.0

# Generate chain
chain = generate_globular_chain(num_points, step_size)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(chain[:, 0], chain[:, 1], chain[:, 2])
ax.scatter(chain[:, 0], chain[:, 1], chain[:, 2], c='r', marker='o')
plt.show()

