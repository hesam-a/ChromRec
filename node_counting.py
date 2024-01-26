import sys
import numpy as np

def parse_pdb(file_path):
    atoms = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                parts = line.split()
                x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
                atoms.append((x, y, z))
    return atoms


def get_bounds(atoms, padding=5.0):
    # Convert to numpy array for easier manipulation
    atom_coords = np.array(atoms)

    # Find min and max along each dimension with some padding
    min_bounds = np.min(atom_coords, axis=0) - padding
    max_bounds = np.max(atom_coords, axis=0) + padding

    return min_bounds, max_bounds


def create_grid(min_bounds, max_bounds, step=0.5):
    # Generate a grid of points
    x_range = np.arange(min_bounds[0], max_bounds[0], step)
    y_range = np.arange(min_bounds[1], max_bounds[1], step)
    z_range = np.arange(min_bounds[2], max_bounds[2], step)

    grid = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)
    return grid


def is_inside_protein(node, atoms, threshold=2):
    # Find the closest atom to the node
    closest_atom_distance = np.min(np.linalg.norm(np.array(atoms) - node, axis=1))
    return closest_atom_distance < threshold


def count_nodes_inside_protein(grid, atoms, threshold=2):
    count = 0
    for node in grid:
        if is_inside_protein(node, atoms, threshold):
            count += 1
    return count


step = 0.5
atoms = parse_pdb(sys.argv[1])
min_bounds, max_bounds = get_bounds(atoms)
grid = create_grid(min_bounds, max_bounds, step)
node_count = count_nodes_inside_protein(grid, atoms)

# Calculate volume 
volume_per_cell = step ** 3  # Step is the grid step size
total_volume = node_count * volume_per_cell
print(f"Approximate Volume: {total_volume}")
