import numpy as np
import random
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# === Provided Helper Functions (unchanged) ===

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
        new_particle = exit_dna + i * 40 * direction  # 40 Å per DNA particle
        new_particles = np.vstack([new_particles, new_particle])
    return new_particles

def rand_gen(mu, stddev, items):
    gaussian_samples = np.random.normal(mu, stddev, 1000000)
    gaussian_samples = np.clip(gaussian_samples, 0, len(items) - 1)
    indices = np.round(gaussian_samples).astype(int)
    unique, counts = np.unique(indices, return_counts=True)
    pmf = counts / sum(counts)
    probabilities = {item: pmf[i] for i, item in enumerate(unique)}
    item_probabilities = [probabilities.get(item, 0) for item in items]
    item_probabilities = np.array(item_probabilities) / np.sum(item_probabilities)
    return item_probabilities

def compute_acetylation_pattern(loop_nucs, acetylation_ratio):
    """
    Compute indices of nucleosomes that should be acetylated in a loop.
    If the computed number of acetylated nucleosomes is zero (due to a low ratio and/or small loop size),
    an empty set is returned.
    """
    num_acetylated = int(loop_nucs * acetylation_ratio)
    if num_acetylated <= 0:
        return set()  # No acetylation for this loop.
    step_size = loop_nucs // num_acetylated  # Even spacing.
    # Ensure that the step_size is at least 1 (it normally is if num_acetylated > 0)
    step_size = max(1, step_size)
    acetylation_indices = set(range(0, loop_nucs, step_size))
    return acetylation_indices

def box_side_length(num_points, occupancy_percentage):
    single_nuc_vol = 6.97e-25  # in m³
    volume_nucleosomes_total = num_points * single_nuc_vol
    required_box_volume = volume_nucleosomes_total / occupancy_percentage
    box_side = round(required_box_volume ** (1 / 3), 8)
    return box_side * 1e10  # in angstroms

def write_lammps_data_file(num_points, occupancy_percentage, atoms, bonds, angles):
    with open("chromatin_system.data", "w") as f:
        box_side = box_side_length(num_points, occupancy_percentage) + 100 
        xmin = ymin = zmin = -box_side/2
        xmax = ymax = zmax = box_side/2
        f.write("# LAMMPS data file\n\n")
        f.write(f"   {len(atoms)}   atoms\n")
        f.write(f"   {len(bonds)}   bonds\n")
        f.write(f"   {len(angles)}   angles\n\n")
        f.write("   10    atom types \n")
        f.write("   44   bond types \n")
        f.write("   2    angle types\n\n")
        f.write(f"   {str(round(xmin, 2))}  {str(round(xmax, 2))} xlo xhi\n")
        f.write(f"   {str(round(ymin, 2))}  {str(round(ymax, 2))} ylo yhi\n")
        f.write(f"   {str(round(zmin, 2))}  {str(round(zmax, 2))} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("   1    15273  # H3 \n")
        f.write("   2    11390  # H4 \n")
        f.write("   3    13789  # H2A\n")
        f.write("   4    13990  # H2B\n")
        f.write("   5    9267   # DNA\n")
        f.write("   6    2379   # H3 tail ARTKQTARKSTGGKAPRKL\n")
        f.write("   7    723    # H4 tail KRHRVKVLR          \n")
        f.write("   8    1483   # Acidic Patch               \n")
        f.write("   9    2379   # Acetylated H3 Tail\n")
        f.write("   10   723    # Acetylated H4 Tail\n")
        f.write("\nAtoms\n\n")
        for atom in atoms:
            f.write(atom)
        f.write("\nBonds\n\n")
        for bond in bonds:
            f.write(bond)
        f.write("\nAngles\n\n")
        for angle in angles:
            f.write(angle)

# === New Helper Functions for the Full Nucleosome Chain Generator ===

def initialize_points(start_point):
    """Return a list with the starting point (or default origin)."""
    if start_point is not None:
        return [np.array(start_point)]
    else:
        return [np.array([0.0, 0.0, 0.0])]


#def is_in_bounds(points, bounds):
#    """
#    Check if all points are within the provided bounds (with a margin).
#    points: A NumPy array of shape (N, 3) (multiple points) or (3,) (single point).
#    bounds: Tuple (xmin, xmax, ymin, ymax, zmin, zmax).
#
#    Returns:
#    - A single boolean (True if all points are in bounds, False otherwise).
#    """
#    margin = 0.3 * bounds[1]  # Takes care of COM of nucs at the wall with long linker DNA
#    xmin, xmax, ymin, ymax, zmin, zmax = bounds
#
#    points = np.atleast_2d(points)  # Ensure the input is 2D
#
#    x_in_bounds = (xmin + margin <= points[:, 0]) & (points[:, 0] <= xmax - margin)
#    y_in_bounds = (ymin + margin <= points[:, 1]) & (points[:, 1] <= ymax - margin)
#    z_in_bounds = (zmin + margin <= points[:, 2]) & (points[:, 2] <= zmax - margin)
#
#    result = x_in_bounds & y_in_bounds & z_in_bounds  # Boolean array
#
#    return np.all(result)  # Return a single True/False value


def is_in_bounds(point, bounds):
    """
    Check that the point is within the provided bounds (with a margin).
    bounds is a tuple: (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    margin = 0.3 * bounds[1] # takes care of COM of nucs at the wall with long linker DNA
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    return (xmin + margin <= point[0] <= xmax - margin and
            ymin + margin <= point[1] <= ymax - margin and
            zmin + margin <= point[2] <= zmax - margin)

def generate_next_point(points, step_size, center_bias, bounds, kd_tree, radial_distance, max_attempts):
    """
    Try to generate a valid new point using a random step, center bias, and checking both bounds and distance.
    Returns (new_point, updated_kd_tree, attempts, found_valid)
    """

    attempts = 0
    new_point = None
    found_valid = False
    while not found_valid and attempts < max_attempts:
        attempts += 1
        # Generate a random step
        step = np.random.normal(size=3)
        step = step / np.linalg.norm(step) * step_size
        candidate = points[-1] + step
        # Apply bias toward the center of mass
        center_of_mass = np.mean(points, axis=0)
        direction_to_center = center_of_mass - candidate
        candidate += center_bias * direction_to_center
        # Check bounds
        if is_in_bounds(candidate, bounds):
            not_too_close = True
            if kd_tree is not None:
                min_dist = kd_tree.query(candidate, k=1)[0]
                if min_dist < radial_distance:
                    not_too_close = False
            # If no kd_tree exists, assume valid.
            if not_too_close:
                found_valid = True
                new_point = candidate
                # Update the KDTree with all current points + candidate
                updated_points = np.array(points + [candidate])
                kd_tree = cKDTree(updated_points)

    #print(kd_tree.data.shape)  
    #print(kd_tree.data)
    return new_point, kd_tree, attempts, found_valid


def perform_backtracking(points, atoms, bonds, angles,
                         atoms_per_nucleosome, bonds_per_nucleosome, angles_per_nucleosome,
                         backtrack_steps, atom_idx_start, mol_idx_start, bond_idx_start, angle_idx_start, natoms):
    """
    Backtrack by removing the last backtrack_steps nucleosome entries from the points, atoms,
    bonds, and angles lists and update the indices accordingly.

    This implementation exactly follows the older logic:

      - Remove up to backtrack_steps points (but always leave at least one point).
      - For each nucleosome removed:
          * Decrement mol_idx_start by 1.
          * Remove the number of atoms stored in atoms_per_nucleosome.
          * Remove the number of bonds stored in bonds_per_nucleosome.
          * Remove the number of angles stored in angles_per_nucleosome.
          * Adjust natoms, atom_idx_start, bond_idx_start, and angle_idx_start accordingly.
    """
    # Remove up to backtrack_steps points (ensuring at least one remains)
    for _ in range(min(backtrack_steps, len(points) - 1)):
        points.pop()

    # Backtrack atoms: For each nucleosome removed, remove its atoms.
    for _ in range(backtrack_steps):
        if len(atoms_per_nucleosome) > 0:
            # Decrement the nucleosome (molecule) counter
            mol_idx_start -= 1
            num_atoms_to_remove = atoms_per_nucleosome.pop()
            for _ in range(num_atoms_to_remove):
                atoms.pop()
            natoms -= num_atoms_to_remove
            atom_idx_start -= num_atoms_to_remove

    # Backtrack bonds: Remove bonds for each nucleosome removed.
    for _ in range(backtrack_steps):
        if len(bonds_per_nucleosome) > 0:
            num_bonds_to_remove = bonds_per_nucleosome.pop()
            for _ in range(num_bonds_to_remove):
                bonds.pop()
            bond_idx_start -= num_bonds_to_remove

    # Backtrack angles: Remove angles for each nucleosome removed.
    for _ in range(backtrack_steps):
        if len(angles_per_nucleosome) > 0:
            num_angles_to_remove = angles_per_nucleosome.pop()
            for _ in range(num_angles_to_remove):
                angles.pop()
            angle_idx_start -= num_angles_to_remove

    return points, atoms, bonds, angles, atom_idx_start, mol_idx_start, bond_idx_start, angle_idx_start, natoms


def apply_random_rotations(current_particles, angle_d, angle_d2):
    """
    Apply the three sequential rotations to the current nucleosome.
    Returns the rotated particles.
    """
    # First random angle
    theta1 = (np.pi + random.uniform(-angle_d, angle_d)) * random.choice([-1, 1])
    # Second and third angles
    theta2 = (np.deg2rad(30) + random.uniform(-angle_d, angle_d)) * random.choice([-1, 1])
    theta3 = (np.deg2rad(45) + random.uniform(-angle_d2, angle_d2)) * random.choice([-1, 1])
    
    # First rotation
    axis1 = dir_vec(current_particles[16], current_particles[18])
    R1 = rotation_matrix(axis1, theta1)
    rotated_particles = rotate_particles(current_particles, R1, current_particles[14])
    
    # Second (in-plane) rotation
    fixed_point = rotated_particles[18]
    v1 = rotated_particles[16] - fixed_point
    v2 = rotated_particles[20] - fixed_point
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    R2 = in_plane_rotation_matrix(theta2, normal)
    rotated_nuc = rotate_particles(rotated_particles, R2, fixed_point)
    
    # Third rotation with fixed particles (e.g., indices 25 and 26 remain fixed)
    fixed_indices = [25, 26]
    rotated_plane = rotate_particles_fix_three(rotated_nuc, fixed_indices, theta3)
    return rotated_plane

def add_linkers_and_translate(particles, items, pmf, target_point, last_iteration=False):
    """
    Add a random number of linker DNA particles (using pmf) and translate the entire set
    so that its center of mass matches the target point.

    If last_iteration is True, no linker DNA is added.

    Returns:
        translated_particles: the translated (and possibly augmented) particle array
        num_linkers: number of linker DNA particles added (zero if last_iteration is True)
    """
    if last_iteration:
        # Do not add linker DNA on the last iteration.
        translated_particles = particles.copy()
        num_linkers = 0
    else:
        num_linkers = np.random.choice(items, p=pmf)
        particles_with_linkers = add_linker_dna(particles, num_linkers)
        center_of_mass = np.mean(particles_with_linkers, axis=0)
        translation_vector = target_point - center_of_mass
        translated_particles = particles_with_linkers + translation_vector
    return translated_particles, num_linkers

def assemble_atoms(translated_particles, atom_idx_start, mol_idx, tail_type_h3, tail_type_h4, atom_comment):
    """
    Assemble formatted atom strings for the given nucleosome particles.
    Returns the list of atom strings and the updated atom index.
    """
    atoms = []
    for j in range(len(translated_particles)):
        if j < 8:
            atom_type = (j % 4) + 1
        elif 8 <= j < 10:
            atom_type = tail_type_h3
        elif 10 <= j < 12:
            atom_type = tail_type_h4
        elif 12 <= j < 14:
            atom_type = 8
        else:
            atom_type = 5
        atoms.append(f"   {atom_idx_start+j:<10}{mol_idx:<8}{atom_type:<8}"
                     f"{translated_particles[j][0]:<15.5f}{translated_particles[j][1]:<15.5f}"
                     f"{translated_particles[j][2]:<15.5f}{atom_comment[j]}\n")
    return atoms, atom_idx_start + len(translated_particles)

def assemble_bonds(natoms, bond_idx_start, original_bonds):
    """
    Assemble bond strings based on the provided original bond definitions.
    Returns the list of bond strings and the updated bond index.
    """
    bonds = []
    for bond_idx, bond in enumerate(original_bonds):
        bonds.append(f"   {bond_idx_start + bond_idx:<10}{int(bond[0]):<8}"
                     f"{int(bond[1]) + natoms:<10}{int(bond[2]) + natoms:<10}{bond[3]:<3}\n")
    return bonds, bond_idx_start + len(original_bonds)

def assemble_angles(natoms, angle_idx_start, original_angles):
    """
    Assemble angle strings based on the provided original angle definitions.
    Returns the list of angle strings and the updated angle index.
    """
    angles_local = []
    for angle_idx, angle in enumerate(original_angles):
        # Remove the extra "+1" offsets so that the atom IDs remain within the valid range.
        angles_local.append(f"   {angle_idx_start + angle_idx:<10}{int(angle[0]):<8}"
                            f"{int(angle[1]) + natoms:<10}{int(angle[2]) + natoms:<10}"
                            f"{int(angle[3]) + natoms:<10}{angle[4]:<3}\n")
    return angles_local, angle_idx_start + len(original_angles)


def assemble_linker_connections(natoms, atom_idx_start, bond_idx_start, angle_idx_start,
                                linker_DNA_bonds, linker_DNA_angles, num_linkers,
                                last_iteration, i, num_points):
    """
    Assemble bonds/angles for linker DNA and connecting bonds/angles between nucleosomes.
    Returns lists of bond and angle strings and updated bond and angle indices.
    """
    bonds = []
    angles = []
    if i < num_points or not last_iteration:
        for idx in range(num_linkers):
            bonds.append(f"   {bond_idx_start:<10}{int(linker_DNA_bonds[idx][0]):<8}"
                         f"{int(linker_DNA_bonds[idx][1]) + natoms:<10}"
                         f"{int(linker_DNA_bonds[idx][2]) + natoms:<10}"
                         f"{linker_DNA_bonds[idx][3]:<3} - Linker DNAs\n")
            bond_idx_start += 1
            angles.append(f"   {angle_idx_start:<10}{int(linker_DNA_angles[idx][0]):<8}"
                          f"{int(linker_DNA_angles[idx][1]) + natoms + 1:<10}"
                          f"{int(linker_DNA_angles[idx][2]) + natoms + 1:<10}"
                          f"{int(linker_DNA_angles[idx][3]) + natoms + 1:<10}"
                          f"{linker_DNA_angles[idx][4]:<3} - Linker DNAs\n")
            angle_idx_start += 1
        # Add connecting bonds/angles between nucleosomes:
        bonds.append(f"   {bond_idx_start:<10}{str(14):<8}{atom_idx_start-1:<10}{atom_idx_start+14:<10}   #  Connecting DNAs\n")
        bond_idx_start += 1
        angles.append(f"   {angle_idx_start:<10}{str(1):<8}{atom_idx_start-2:<10}{atom_idx_start-1:<10}{atom_idx_start+14:<10}#  Connecting DNAs\n")
        angle_idx_start += 1
        angles.append(f"   {angle_idx_start:<10}{str(1):<8}{atom_idx_start-1:<10}{atom_idx_start+14:<10}{atom_idx_start+15:<10}#  Connecting DNAs\n")
        angle_idx_start += 1
    return bonds, angles, bond_idx_start, angle_idx_start

# === Main Function: generate_full_nucleosome_chain ===

def generate_full_nucleosome_chain(nuc, num_points, total_nucs, step_size, radial_distance, acetylation_ratio,
                                   occupancy_percentage, acetylate=False, kd_tree=None, start_point=None,
                                   atom_idx_start=1, mol_idx_start=1, bond_idx_start=1, angle_idx_start=1,
                                   atom_comment=None, original_bonds=None, original_angles=None,
                                   linker_DNA_bonds=None, linker_DNA_angles=None, last_iteration=False,
                                   max_attempts=1200, backtrack_steps=5, center_bias=0.01):

    """
    Generate a full nucleosome chain with (optionally) alternating acetylation.
    Returns a dictionary with atom, bond, and angle entries and updated indices.
    """
    # Initialize the starting point and define simulation bounds
    points = initialize_points(start_point)
    box_side = box_side_length(total_nucs, occupancy_percentage) + 100
    bounds = (-box_side/2, box_side/2, -box_side/2, box_side/2, -box_side/2, box_side/2)
    current_particles = nuc.copy()
    
    # Determine acetylation pattern if needed
    if acetylate:
        acetylation_indices = compute_acetylation_pattern(num_points, acetylation_ratio)
    
    angle_d = np.deg2rad(10)
    angle_d2 = np.deg2rad(20)
    
    # Set up probability distribution for the number of linker DNA particles.
    items = [0, 1, 2, 3, 4, 5]
    mu = 2.5
    sigma = 1.35
    pmf = rand_gen(mu, sigma, items)
    
    # Initialize accumulators and counters.
    all_atoms = []
    all_bonds = []
    all_angles = []
    atoms_per_nucleosome = []   # To help with backtracking
    bonds_per_nucleosome = []
    angles_per_nucleosome = []
    natoms = atom_idx_start - 1  # Total atoms so far

    i = 1
    while i <= num_points:
        found_valid = False
        attempts = 0

        # Decide acetylation for this nucleosome if needed.
        if acetylate:
            acetylated = (i in acetylation_indices)
            tail_type_h3 = 9 if acetylated else 6
            tail_type_h4 = 10 if acetylated else 7
        else:
            tail_type_h3 = 6
            tail_type_h4 = 7

        # Try to generate a new valid point.
        new_point, kd_tree, attempts, found_valid = generate_next_point(
            points, step_size, center_bias, bounds, kd_tree, radial_distance, max_attempts)
        
        if found_valid:
            points.append(new_point)
            
            # Rotate the nucleosome.
            rotated_particles = apply_random_rotations(current_particles, angle_d, angle_d2)
            
            # Add random linker DNA and translate so that the nucleosome’s COM meets new_point.
            translated_particles, num_linkers = add_linkers_and_translate(rotated_particles, items, pmf, points[-1], last_iteration)
            
            # Update total atom count (natoms) before adding this nucleosome.
            natoms = atom_idx_start - 1
            
            # Assemble atoms, bonds, and angles for this nucleosome.
            nucleosome_atoms, atom_idx_start = assemble_atoms(translated_particles, atom_idx_start, mol_idx_start, tail_type_h3, tail_type_h4, atom_comment)
    
            nucleosome_bonds, bond_idx_start = assemble_bonds(natoms, bond_idx_start, original_bonds)
            nucleosome_angles, angle_idx_start = assemble_angles(natoms, angle_idx_start, original_angles)
            
            # Assemble linker DNA connections if this isn’t the final nucleosome.
            linker_bonds, linker_angles, bond_idx_start, angle_idx_start = assemble_linker_connections(
                natoms, atom_idx_start, bond_idx_start, angle_idx_start, linker_DNA_bonds,
                linker_DNA_angles, num_linkers, last_iteration, i, num_points)
            
            # Append to the overall lists.
            all_atoms.extend(nucleosome_atoms)
            all_bonds.extend(nucleosome_bonds)
            all_bonds.extend(linker_bonds)
            all_angles.extend(nucleosome_angles)
            all_angles.extend(linker_angles)
            
            # Record counts for potential backtracking.
            atoms_per_nucleosome.append(len(translated_particles))
            bonds_per_nucleosome.append(len(original_bonds) + num_linkers + 1)
            angles_per_nucleosome.append(len(original_angles) + num_linkers + 2)
            
            mol_idx_start += 1
            # Set the current_particles for the next nucleosome.
            current_particles = rotated_particles
            
            #print(f"Generated nucleosome {i}/{num_points}")
            i += 1
        else:
            
            # If backtracking is needed
            print(f"Backtracking from nucleosome {i} after {attempts} failed attempts")
            print(f"Before backtracking: Total atoms: {natoms}, Bonds: {len(all_bonds)}, Angles: {len(all_angles)}")
            print(f"Indices before backtracking: Atom index: {atom_idx_start}, Bond index: {bond_idx_start}, Angle index: {angle_idx_start}")
        
            (points, all_atoms, all_bonds, all_angles, atom_idx_start, mol_idx_start,
                    bond_idx_start, angle_idx_start, natoms) = perform_backtracking(points, all_atoms, all_bonds, all_angles,
                    atoms_per_nucleosome, bonds_per_nucleosome, angles_per_nucleosome, backtrack_steps, atom_idx_start,
                    mol_idx_start, bond_idx_start, angle_idx_start, natoms)
            print(f"After backtracking: Total atoms: {natoms}, Bonds: {len(all_bonds)}, Angles: {len(all_angles)}")
            print(f"Updated indices after backtracking: Atom index: {atom_idx_start}, Bond index: {bond_idx_start}, Angle index: {angle_idx_start}")
            i = max(1, i - backtrack_steps)
    
    # (Optionally) write an XYZ file for visualization.
    with open(f"chromatin_{num_points}.xyz", 'w') as g:
        g.write(f"{num_points+1}\n\n")
        for point in points:
            g.write(f"   C   {point[0]:<15.5f}{point[1]:<15.5f}{point[2]:<15.5f}\n")
    
    return {
        "atoms": all_atoms,
        "bonds": all_bonds,
        "angles": all_angles,
        "atom_idx_start": atom_idx_start,
        "mol_idx_start": mol_idx_start,
        "bond_idx_start": bond_idx_start,
        "angle_idx_start": angle_idx_start,
    }



def generate_chromatin_with_loops(nuc, total_nucs, loop_nucs, nlinking_nucs, step_size, radial_distance, occupancy_percentage,
                                  high_acetylation=0.8, low_acetylation=0.05,
                                  kd_tree=None, atom_comment=None, original_bonds=None, original_angles=None,
                                  linker_DNA_bonds=None, linker_DNA_angles=None):
    """
    Generates a chromatin structure consisting of multiple loops of nucleosomes.
    Each loop is generated as a chain using generate_full_nucleosome_chain.
    Linking nucleosomes (which connect two loops) are generated between loops.
    
    The following state is tracked between loops:
      - last_dna_position: the last generated DNA point (so that DNA is continuous)
      - all_atoms, all_bonds, all_angles: accumulated LAMMPS entry strings
      - atom_idx_start, mol_idx_start, bond_idx_start, angle_idx_start: running indices for LAMMPS entries
      - kd_tree: used for spatial overlap checking
    
    Parameters:
      nuc                  : The nucleosome particle array (or template)
      total_nucs           : Total number of nucleosomes in the entire structure
      loop_nucs            : Number of nucleosomes per loop
      nlinking_nucs        : Number of nucleosomes to use for linking adjacent loops
      step_size            : Step size for random nucleosome placement
      radial_distance      : Minimum separation between nucleosome positions
      occupancy_percentage : Percentage occupancy used to calculate simulation box size
      high_acetylation     : Acetylation ratio for even-numbered loops
      low_acetylation      : Acetylation ratio for odd-numbered loops
      kd_tree              : (Optional) initial KDTree for overlap checks
      atom_comment         : List of comments for each atom (used when assembling atoms)
      original_bonds       : Bond definitions for one nucleosome
      original_angles      : Angle definitions for one nucleosome
      linker_DNA_bonds     : Bond definitions for linker DNA segments
      linker_DNA_angles    : Angle definitions for linker DNA segments
      
    Returns:
      Nothing directly. This function writes out the final LAMMPS data file and returns the final state
      (if needed) as a dictionary.
    """
    # Determine the number of loops
    num_loops = total_nucs // loop_nucs
    print(f"\n *** Total number of loops: {num_loops}\n")

    # Initial values to track continuity:
    last_dna_position = None   # Will be updated from the chain generator's last point
    all_atoms = []
    all_bonds = []
    all_angles = []
    # Start indices for atoms, molecules, bonds, and angles:
    atom_idx_start = 1
    mol_idx_start = 1
    bond_idx_start = 1
    angle_idx_start = 1

    # Loop over the number of loops.
    for i in range(num_loops):
        # Flag for the last loop:
        last_iteration = (i == num_loops - 1)
        # Alternate the acetylation ratio between loops:
        acetylation_ratio = high_acetylation if i % 2 == 0 else low_acetylation

        # Generate one full nucleosome chain (i.e. one loop)
        chain_data = generate_full_nucleosome_chain(
            nuc,
            loop_nucs,                # total nucleosomes in this loop
            total_nucs,
            step_size,
            radial_distance,
            acetylation_ratio,
            occupancy_percentage,
            True,                     # acetylate = True (using full nucleosome features)
            kd_tree,
            last_dna_position,        # start_point is the last DNA position from previous loop (or None for first loop)
            atom_idx_start,
            mol_idx_start,
            bond_idx_start,
            angle_idx_start,
            atom_comment,
            original_bonds,
            original_angles,
            linker_DNA_bonds,
            linker_DNA_angles,
            last_iteration=last_iteration,
            max_attempts=1200,
            backtrack_steps=5,
            center_bias=0.01
        )
        # Update the current state using the returned values:
        all_atoms.extend(chain_data["atoms"])
        last_dna_position = np.array(all_atoms[-1].split()[3:6], dtype=float)
        all_bonds.extend(chain_data["bonds"])
        all_angles.extend(chain_data["angles"])
        atom_idx_start = chain_data["atom_idx_start"]
        mol_idx_start = chain_data["mol_idx_start"]
        bond_idx_start = chain_data["bond_idx_start"]
        angle_idx_start = chain_data["angle_idx_start"]

        print(f"   *** LOOP {i+1} was generated!!! ***")

        # If this is not the last loop, generate linking nucleosomes between loops.
        if not last_iteration:
            # For linking nucleosomes we use the same generator but with acetylate=False.
            linking_data = generate_full_nucleosome_chain(
                nuc,
                nlinking_nucs,           # use the linking nucleosome count
                total_nucs,
                step_size*1.3,
                radial_distance*1.5,
                0.0,                     # acetylation ratio is set to 0 (or ignored) for linking nucleosomes
                occupancy_percentage,
                False,                   # no acetylation features for linking nucleosomes
                kd_tree,
                last_dna_position,       # continue from the current last position
                atom_idx_start,
                mol_idx_start,
                bond_idx_start,
                angle_idx_start,
                atom_comment,
                original_bonds,
                original_angles,
                linker_DNA_bonds,
                linker_DNA_angles,
                last_iteration=False,
                max_attempts=1200,
                backtrack_steps=5,
                center_bias=0.01
            )
            # Update state with linking nucleosomes data:
            all_atoms.extend(linking_data["atoms"])
            all_bonds.extend(linking_data["bonds"])
            all_angles.extend(linking_data["angles"])
            atom_idx_start = linking_data["atom_idx_start"]
            mol_idx_start = linking_data["mol_idx_start"]
            bond_idx_start = linking_data["bond_idx_start"]
            angle_idx_start = linking_data["angle_idx_start"]

            # Update the start position based on the linking nucleosomes:
            last_dna_position = np.array(all_atoms[-1].split()[3:6], dtype=float)
            
            print(f"   ***  A set of {nlinking_nucs} linking nucleoesomes was generated!!! ***")

    # Write the final LAMMPS data file.
    write_lammps_data_file(total_nucs, occupancy_percentage, all_atoms, all_bonds, all_angles)

    # return the final state
    return {
        "last_dna_position": last_dna_position,
        "atoms": all_atoms
        }
    #    "bonds": all_bonds,
    #    "angles": all_angles,
    #    "atom_idx_start": atom_idx_start,
    #    "mol_idx_start": mol_idx_start,
    #    "bond_idx_start": bond_idx_start,
    #    "angle_idx_start": angle_idx_start
    #}


nuc = np.array([
    [131.212189, 126.855835, 137.335938],  #  H3
    [129.095459, 116.979622, 136.993576],  #  H4
    [103.079391, 105.861420, 119.854347],  #  H2A
    [109.069916,  98.158401, 119.412384],  #  H2B
    [107.114456, 133.206940, 110.713539],  #  H3
    [112.637611, 127.008202, 104.648575],  #  H4
    [142.811447, 116.968361, 112.810295],  #  H2A
    [140.466858, 110.083466, 106.688095],  #  H2B
    [159.257395, 131.786325, 155.914330],  #  H3AT
    [ 79.069249, 128.276449,  92.135146],  #  H3ET
    [145.620564, 104.780081, 156.750653],  #  H4BT
    [ 98.557190, 127.123703,  79.869987],  #  H4FT
    [106.095863, 105.601272, 124.436172],  #  ACPTCH1
    [139.864410, 118.947700, 108.967658],  #  ACPTCH2
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
])


dna_indices = list(range(8, 23))


# Define the bond and angle types from the first nucleosome
original_bonds = np.array([
     [1, 1, 2, "#  A - B"],
     [1, 5, 6, "#  E - F"],
     [2, 3, 4, "#  C - D"],
     [2, 7, 8, "#  G - H"],
     [3, 1, 5, "#  A - E"],
     [4, 2, 6, "#  B - F"],
     [5, 3, 7, "#  C - G"],
     [6, 4, 8, "#  D - H"],
     [7, 2, 4, "#  B - D"],
     [7, 2, 8, "#  B - H"],
     [7, 6, 4, "#  F - D"],
     [7, 6, 8, "#  F - H"],
     [8, 1, 7, "#  A - G"],
     [8, 5, 3, "#  E - C"],
     [9, 5, 7, "#  E - G"],
     [9, 1, 3, "#  A - C"],
     [10, 1, 8, "#  H3-H2B 1-8 36.1"],
     [10, 5, 4, "#  H3-H2B 5-4 36.1"],
     [11, 5, 8, "#  H3-H2B 5-8 40.5"],
     [11, 1, 4, "#  H3-H2B 1-4 40.5"],
     [12, 2, 7, "#  H4-H2A 2-7 27.8"],
     [12, 6, 3, "#  H4-H2A 6-3 27.8"],
     [13, 2, 3, "#  H4-H2A 2-3 32.9"],
     [13, 6, 7, "#  H4-H2A 6-7 32.9"],
     [14, 15, 16, '#  DNA'],
     [14, 16, 17, '#  DNA'],
     [14, 17, 18, '#  DNA'],
     [14, 18, 19, '#  DNA'],
     [14, 19, 20, '#  DNA'],
     [14, 20, 21, '#  DNA'],
     [14, 21, 22, '#  DNA'],
     [14, 22, 23, '#  DNA'],
     [14, 23, 24, '#  DNA'],
     [14, 24, 25, '#  DNA'],
     [14, 25, 26, '#  DNA'],
     [14, 26, 27, '#  DNA'],
     [15, 16, 5, '#  DNA - H3 30.2'],
     [16, 16, 3, '#  DNA - H2A 34.1'],
     [17, 17, 3, '#  DNA - H2A 24.0'],
     [18, 17, 6, '#  DNA - H2B 38.5'],
     [19, 18, 4, '#  DNA - H2B 27.2'],
     [20, 18, 8, '#  DNA - H2B 38.1'],
     [21, 19, 2, '#  DNA - H4 25.0'],
     [22, 19, 7, '#  DNA - H2A 37.2'],
     [23, 20, 1, '#  DNA - H3 19.0'],
     [24, 20, 7, '#  DNA - H4 45.0'],
     [25, 21, 1, '#  DNA - H3 37.2'],
     [26, 21, 5, '#  DNA - H3 28.0'],
     [27, 22, 3, '#  DNA - H3 42.6'],
     [28, 22, 5, '#  DNA - H4 21.8'],
     [29, 23, 6, '#  DNA - H4 30.9'],
     [30, 23, 4, '#  DNA - H2B 37.1'],
     [31, 24, 8, '#  DNA - H2B 23.0'],
     [32, 24, 4, '#  DNA - H3 40.0'],
     [33, 25, 1, '#  DNA - H3 35.7'],
     [34, 25, 7, '#  DNA - H2A 25.4'],
     [35, 26, 7, '#  DNA - H2A 44.1'],
     [36, 26, 1, '#  DNA - H3 34.3'],
     [37, 15, 21, '#  DNA 39.6'],
     [38, 21, 27, '#  DNA 44.1'],
     [39, 16, 21, '#  DNA 43.9'],
     [40, 21, 26, '#  DNA 47.2'],
     [41,  1, 9, '#  H3A - tail'],
     [41,  5, 10,'#  H3E - tail'],
     [42,  2, 11,'#  H4B - tail'],
     [42,  6, 12,'#  H4F - tail'],
     [43,  3, 13,'#  H2A - ACPTCH1'],
     [44,  4, 13,'#  H2B - ACPTCH1'],
     [43,  7, 14,'#  H2A - ACPTCH2'],
     [44,  8, 14,'#  H2B - ACPTCH2'],
     ])


linker_DNA_bonds = np.array([
    [14, 27, 28, "#  DNA13 - DNA14"],
    [14, 28, 29, "#  DNA14 - DNA15"],
    [14, 29, 30, "#  DNA15 - DNA16"],
    [14, 30, 31, "#  DNA16 - DNA17"],
    [14, 31, 32, "#  DNA17 - DNA18"],
    [14, 32, 33, "#  DNA18 - DNA19"]])

original_angles = np.array([
#    [1, 22, 9 , 10], 
    [1, 15, 16, 17, "#  DNA1 - DNA2 - DNA3"],
    [1, 16, 17, 18, "#  DNA2 - DNA3 - DNA4"],
    [1, 17, 18, 19, "#  DNA3 - DNA4 - DNA5"],
    [1, 18, 19, 20, "#  DNA4 - DNA5 - DNA6"],
    [1, 19, 20, 21, "#  DNA5 - DNA6 - DNA7"],
    [1, 20, 21, 22, "#  DNA6 - DNA7 - DNA8"],
    [1, 21, 22, 23, "#  DNA7 - DNA8 - DNA9"],
    [1, 22, 23, 24, "#  DNA8 - DNA9 - DNA10"],
    [1, 23, 24, 25, "#  DNA9 - DNA10 - DNA11"],
    [1, 24, 25, 26, "#  DNA10 - DNA11 - DNA12"],
    [1, 25, 26, 27, "#  DNA11 - DNA12 - DNA13"],
    [2,  3, 13,  4, "#  H2A - ACPTCH1 - H2B"],
    [2,  7, 14,  8, "#  H2A - ACPTCH1 - H2B"]])

#    [2, 9 , 15, 21]])
#   [3, 10, 15, 20]]

linker_DNA_angles = np.array([
    [1, 26, 27, 28, "#  DNA12 - DNA13 - DNA14"],
    [1, 27, 28, 29, "#  DNA13 - DNA14 - DNA15"],
    [1, 28, 29, 30, "#  DNA14 - DNA15 - DNA16"],
    [1, 29, 30, 31, "#  DNA15 - DNA16 - DNA17"],
    [1, 30, 31, 32, "#  DNA16 - DNA17 - DNA18"],
    [1, 31, 32, 33, "#  DNA17 - DNA18 - DNA19"],
    ])

atom_comment = [" #  H3 ",   " #  H4  ",  " #  H2A",   " #  H2B",   " #  H3",    " #  H4",      " #  H2A",
                " #  H2B",   " #  H3AT",  " #  H3ET",  " #  H4BT",  " #  H4FT", " #  ACPTCH1", " #  ACPTCH2",
                " #  DNA1",  " #  DNA2",  " #  DNA3",  " #  DNA4",  " #  DNA5",  " #  DNA6",    " #  DNA7",
                " #  DNA8",  " #  DNA9",  " #  DNA10", " #  DNA11", " #  DNA12", " #  DNA13",   " #  DNA14",
                " #  DNA15", " #  DNA16", " #  DNA17", " #  DNA18", " #  DNA19", " #  DNA20"]



# Parameters
loop_nucs = 1000
nlinking_nucs = 10 
total_nucs = 1000
step_size = 120.
radial_distance = 100
high_acetylation=0.8
low_acetylation=0.05
occupancy_percentage = 0.4
acetylate = False
kd_tree=None
start_point=None
atom_idx_start=1
mol_idx_start=1
bond_idx_start=1
angle_idx_start=1

acetylation_ratio = 0.8

box_side = box_side_length(total_nucs, occupancy_percentage) + 100
bounds = (-box_side/2, box_side/2, -box_side/2, box_side/2, -box_side/2, box_side/2)

margin = 0.3 * bounds[1] 
print(f" margin: {margin}")

xmin, xmax, ymin, ymax, zmin, zmax = bounds

print(f" box side length: {bounds[1]}")
print(f" xmin + margin: {xmin + margin}\n xmax - margin: {xmax - margin}")

#print(f" min_x {np.min(nuc[:,0])}  max_x {np.max(nuc[:,0])}")
#print(f" min_y {np.min(nuc[:,1])}  max_y {np.max(nuc[:,1])}")
#print(f" min_z {np.min(nuc[:,2])}  max_z {np.max(nuc[:,2])}")
#
#print(f"center_of_mass of nuc: {np.mean(nuc, axis=0)}")

chain = generate_chromatin_with_loops(nuc, total_nucs, loop_nucs, nlinking_nucs, step_size, radial_distance, 
                              occupancy_percentage, high_acetylation, low_acetylation, kd_tree, atom_comment, 
                              original_bonds, original_angles, linker_DNA_bonds, linker_DNA_angles)

#generate_full_nucleosome_chain(nuc, total_nucs, step_size, radial_distance, acetylation_ratio,
#                               occupancy_percentage, acetylate, kd_tree, start_point,
#                               atom_idx_start, mol_idx_start, bond_idx_start, angle_idx_start,
#                               atom_comment, original_bonds,original_angles, linker_DNA_bonds, 
#                               linker_DNA_angles, last_iteration=False, max_attempts=1200, backtrack_steps=5,
#                               center_bias=0.01) 

