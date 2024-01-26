import sys, time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


atomic_radii = {
    "H": 1.2, "C": 1.7,  "N": 1.55,  "O": 1.52, "F": 1.47,
    "P": 1.8, "S": 1.8, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "MN": 1.61,
}

def parse_pdb(file_path):
    atoms = []
    atom_types = []  # Separate list for atom types
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_type = line[76:78].strip().upper()  # Extract and convert atom type to uppercase
                x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                atoms.append((x, y, z))
                atom_types.append(atom_type)
    return atoms, atom_types


def get_bounds(atoms, padding=7.0):
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


def is_inside_protein(node, atoms, atomic_radii, sasa=1.4):
    for x, y, z, atom_type in atoms:
        radius = atomic_radii.get(atom_type, 1.5)  # Default radius if atom type is not in dictionary
        threshold = radius + sasa
        distance = np.linalg.norm(np.array([x, y, z]) - node)
        if distance < threshold:
            return True
    return False


def divide_grid_chunks(grid, num_cpus):
    total_points = len(grid)
    chunk_size = total_points // num_cpus
    chunks = []

    for i in range(num_cpus):
        start_index = i * chunk_size
        # For the last chunk, take all remaining points
        if i == num_cpus - 1:
            end_index = total_points
        else:
            end_index = start_index + chunk_size
        chunks.append(grid[start_index:end_index])

    return chunks


def process_chunk(chunk, atoms, atomic_radii, sasa=1.4):
    inside_points = []
    for node in chunk:
        if is_inside_protein(node, atoms, atomic_radii, sasa):
            inside_points.append(node)
    return inside_points


def get_cube(x, y, z, size):
    """ Generate vertices for a cube starting at (x, y, z). """
    return [
        [(x, y, z), (x+size, y, z), (x+size, y+size, z), (x, y+size, z)],
        [(x, y, z+size), (x+size, y, z+size), (x+size, y+size, z+size), (x, y+size, z+size)],
        [(x, y, z), (x+size, y, z), (x+size, y, z+size), (x, y, z+size)],
        [(x, y+size, z), (x+size, y+size, z), (x+size, y+size, z+size), (x, y+size, z+size)],
        [(x, y, z), (x, y+size, z), (x, y+size, z+size), (x, y, z+size)],
        [(x+size, y, z), (x+size, y+size, z), (x+size, y+size, z+size), (x+size, y, z+size)],
    ]

# Start timing
start_time = time.time()

step = 0.8
sasa = 1.4
num_processes = 4
atoms, atom_types = parse_pdb(sys.argv[1])
combined_atoms = [(x, y, z, atom_type) for ((x, y, z), atom_type) in zip(atoms, atom_types)]
min_bounds, max_bounds = get_bounds(atoms)
grid = create_grid(min_bounds, max_bounds, step)

# Divide the grid into chunks
grid_chunks = divide_grid_chunks(grid, num_processes)

# Process each chunk in parallel using multiprocessing
with Pool(processes=num_processes) as pool:
    args = [(chunk, combined_atoms, atomic_radii, sasa) for chunk in grid_chunks]
    chunk_results = pool.starmap(process_chunk, args)  # Renamed to chunk_results

# Flatten the list of lists
inside_points = [point for sublist in chunk_results for point in sublist]

# Convert to a NumPy array
inside_points = np.array(inside_points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Size of the cube (same as the step size you used for the grid)
cube_size = step

for point in inside_points:
    x, y, z = point
    cube = get_cube(x - cube_size/2, y - cube_size/2, z - cube_size/2, cube_size)
    ax.add_collection3d(Poly3DCollection(cube, facecolors='gray', linewidths=1, edgecolors='black', alpha=0.1))

# End timing
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Total calculation time: {elapsed_time:0.2f} seconds")

ax.set_xlim([min_bounds[0], max_bounds[0]])
ax.set_ylim([min_bounds[1], max_bounds[1]])
ax.set_zlim([min_bounds[2], max_bounds[2]])

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.title('Visualization of Voxels Inside the Protein')

plt.show()
