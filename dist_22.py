import time, sys
import numpy as np
import subprocess, time
from multiprocessing import Pool
from collections import defaultdict

def calculate_distance(atom1, atom2):
    """Calculate the Euclidean distance between two 3D points."""
    return np.linalg.norm(np.array(atom1) - np.array(atom2))

def process_chunk(xyz_file, start_line, end_line, chunk_index, atom_type_dict):
    from collections import defaultdict
    import numpy as np

    # Temporarily store distances before writing to reduce IO operations
    temp_storage_limit = 1000  # Number of distances to accumulate before writing to file
    distances_by_type = defaultdict(list)

    with open(xyz_file, 'r') as file:
        current_line = 0
        for _ in range(start_line):
            next(file)  # Skip lines until start of the chunk
            current_line += 1

        frame_positions, atom_types = [], []
        while current_line < end_line:
            header = file.readline()
            if not header:
                break  # End of file or end of chunk
            num_atoms = int(header.strip())
            file.readline()  # Skip comment line
            current_line += 2  # Account for header and comment line

            frame_positions.clear()
            atom_types.clear()
            for _ in range(num_atoms):
                parts = file.readline().split()
                atom_type = int(parts[0])
                coords = list(map(float, parts[1:4]))
                frame_positions.append(coords)
                atom_types.append(atom_type)
                current_line += 1

            # Calculate distances for all pairs in the frame and categorize by atom type pair
            for i, atom1 in enumerate(frame_positions):
                for j, atom2 in enumerate(frame_positions[i+1:], start=i+1):
                    dist = calculate_distance(atom1, atom2)
                    type_pair = tuple(sorted((atom_type_dict[atom_types[i]], atom_type_dict[atom_types[j]])))
                    distances_by_type[type_pair].append(dist)

                    # Check if we've accumulated enough distances to write to file
                    if len(distances_by_type[type_pair]) >= temp_storage_limit:
                        output_filename = f"chunk_{chunk_index}_{type_pair[0]}_{type_pair[1]}_distances"
                        with open(output_filename, 'a') as outfile:
                            for dist in distances_by_type[type_pair]:
                                outfile.write(f"{dist:.2f}\n")
                        distances_by_type[type_pair].clear()  # Clear the distances from memory

    # After processing the chunk, write any remaining distances to their files
    for type_pair, distances in distances_by_type.items():
        if distances:  # If there are remaining distances to write
            output_filename = f"chunk_{chunk_index}_{type_pair[0]}_{type_pair[1]}_distances"
            with open(output_filename, 'a') as outfile:
                for dist in distances:
                    outfile.write(f"{dist:.2f}\n")

    return chunk_index, [key for key in distances_by_type.keys()]

def get_total_lines(filename):
    result = subprocess.run(['wc', '-l', filename], stdout=subprocess.PIPE)
    return int(result.stdout.split()[0])

def define_chunks(total_lines, lines_per_frame, num_cpus, skip_lines=0):
    """Define chunks for parallel processing."""
    chunk_size = (total_lines - skip_lines) // num_cpus
    chunks = []

    current_start = skip_lines
    while current_start < total_lines:
        current_end = min(current_start + chunk_size, total_lines)
        # Adjust to frame boundary
        if current_end + lines_per_frame <= total_lines:
            current_end += lines_per_frame - (current_end % lines_per_frame)
        chunks.append((current_start, current_end))
        current_start = current_end

    return chunks

atom_type_dict = {1: "H3", 2: "H4", 3: "H2A", 4: "H2B"}

def main():
    start_time = time.time()
    xyz_file = sys.argv[1]
    num_cpus = 4  
    num_particles = 8 
    #n_skip_frames = int(sys.argv[2])

    lines_per_frame = num_particles + 2 
    #skip_lines = n_skip_frames * lines_per_frame

    total_lines = get_total_lines(xyz_file)
    #chunks = define_chunks(total_lines, lines_per_frame, num_cpus, skip_lines)
    chunks = define_chunks(total_lines, lines_per_frame, num_cpus)
    chunk_args = [(xyz_file, start, end, i, atom_type_dict) for i, (start, end) in enumerate(chunks)]

    with Pool(processes=num_cpus) as pool:
        results = pool.starmap(process_chunk, chunk_args)

    end_time = time.time()

    print(f"\n     Measuring histone-histone distances took {(end_time - start_time):.2f} seconds to complete.\n")

if __name__ == "__main__":
    main()
