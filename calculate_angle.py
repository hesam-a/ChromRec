import time, sys
import numpy as np
import subprocess, time
from multiprocessing import Pool
from collections import defaultdict


def calculate_angle(atom1, vertex_atom, atom3):
    """Calculate the angle in degrees between three 3D points where `vertex_atom` is the vertex."""
    vec1 = np.array(atom1) - np.array(vertex_atom)
    vec2 = np.array(atom3) - np.array(vertex_atom)
    dot_product = np.dot(vec1, vec2)
    magnitude_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if magnitude_product == 0:
        return 0  # to handle the division by zero scenario
    angle_rad = np.arccos(dot_product / magnitude_product)
    return np.degrees(angle_rad)


def process_chunk(xyz_file, start_line, end_line, chunk_index, atom_type_dict, angle_types):
    angles_by_type = defaultdict(list)

    with open(xyz_file, 'r') as file:
        current_line = 0
        for _ in range(start_line):
            next(file)
            current_line += 1

        while current_line < end_line:
            # Read header and prepare to process the frame
            header = file.readline()
            num_atoms = int(header.strip())
            file.readline()  # Skip the comment line
            current_line += 2  # Account for header and comment line

            if current_line + num_atoms > end_line:
                break  # Avoid processing partial frames at the end of a chunk

            frame_positions = []
            for _ in range(num_atoms):
                line = file.readline()
                parts = line.split()
                atom_type = int(parts[0])
                coords = list(map(float, parts[1:4]))
                frame_positions.append((atom_type, coords))
                current_line += 1

            # Process each group of 4 atoms
            for i in range(0, num_atoms, 4):  # assuming num_atoms is always a multiple of 4
                group_positions = frame_positions[i:i+4]
                if len(group_positions) < 4:
                    continue  # Skip incomplete groups
                parent_atom = group_positions[0]
                connected_atoms = group_positions[1:]

                # Calculate angles for specified atom triplets
                for atom_triplet, type_key in angle_types.items():
                    atom1_type, vertex_type, atom3_type = map(int, atom_triplet)
                    if parent_atom[0] == vertex_type:
                        atom1 = [atom for atom in connected_atoms if atom[0] == atom1_type]
                        atom3 = [atom for atom in connected_atoms if atom[0] == atom3_type]
                        if atom1 and atom3:
                            angle = calculate_angle(atom1[0][1], parent_atom[1], atom3[0][1])
                            angles_by_type[type_key].append(angle)

    # Write remaining angles to file
    for type_key, angles in angles_by_type.items():
        output_filename = f"chunk_{chunk_index}_angle_{type_key}_angles"
        with open(output_filename, 'a') as outfile:
            for angle in angles:
                outfile.write(f"{angle:.2f}\n")

    return chunk_index, [key for key in angles_by_type.keys()]

def get_total_lines(filename):
    result = subprocess.run(['wc', '-l', filename], stdout=subprocess.PIPE)
    return int(result.stdout.split()[0])

def define_chunks(total_lines, lines_per_frame, num_cpus):
    total_frames = total_lines // lines_per_frame
    num_frames_per_chunk = (total_frames + num_cpus - 1) // num_cpus  # Ceiling division to ensure all frames are processed

    chunks = []
    current_line = 0

    for _ in range(num_cpus):
        start_line = current_line
        end_line = start_line + num_frames_per_chunk * lines_per_frame
        # Ensure the chunk does not exceed the total number of lines
        if end_line > total_lines:
            end_line = total_lines

        chunks.append((start_line, end_line))
        current_line = end_line

        if current_line >= total_lines:
            break  # Stop if we've allocated all lines

    return chunks

def main():
    start_time = time.time()
    #xyz_file = "octamer.xyz"
    xyz_file = sys.argv[1]
    num_cpus = 4
    lines_per_frame = 34  # natoms + 2 lines for header and comment

    angle_types = {
        ('5', '1', '7')  : 1,
        ('5', '1', '13') : 2,
        ('7', '1', '13') : 3,
        ('6', '2', '9')  : 4,
        ('6', '2', '11') : 5,
        ('9', '2', '11') : 6,
        ('8', '3', '10') : 7,
        ('8', '3', '15') : 8,
        ('10', '3', '15'): 9,
        ('14', '4', '12'): 10,
        ('14', '4', '16'): 11,
        ('12', '4', '16'): 12
    }

    atom_type_dict = {1: "H3", 2: "H4", 3: "H2A", 4: "H2B"}


    total_lines = get_total_lines(xyz_file)
    chunks = define_chunks(total_lines, lines_per_frame, num_cpus)

    chunk_args = [(xyz_file, start, end, i, atom_type_dict, angle_types) for i, (start, end) in enumerate(chunks)]

    with Pool(processes=num_cpus) as pool:
        results = pool.starmap(process_chunk, chunk_args)

    end_time = time.time()
    print(f"\nMeasurement of angles completed in {(end_time - start_time):.2f} seconds.\n")

if __name__ == "__main__":
    main()
