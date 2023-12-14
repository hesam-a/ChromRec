import subprocess
import numpy as np
from multiprocessing import Pool
from numba import njit, prange
import logging, csv, sys, subprocess
from itertools import combinations


def extract_last_n_particles_from_xyz(xyz_file, n):
    """
    Extract the positions of the sites that are the last n particles from each frame in an XYZ trajectory.
    """
    frames = []
    with open(xyz_file, 'r') as file:
        while True:
            header = file.readline()
            if not header:
                break  # End of file
            num_atoms = int(header.strip())
            file.readline()  # Skip comment line
            frame_positions = []
            for _ in range(num_atoms - n):
                file.readline()  # Skip comment lines
            for _ in range(n):
                coords = list(map(float, file.readline().split()[1:4]))
                frame_positions.append(coords)
            frames.append(frame_positions)
    return frames


def calculate_distances(frame):
    num_atoms = len(frame)
    distances_matrix = []

    for i in range(num_atoms-1):  # no need to calculate for the last atom
        pos_i = np.array(frame[i])
        distances_row = []
        for j in range(i+1, num_atoms):
            distance = np.linalg.norm(pos_i - np.array(frame[j]))
            distances_row.append(distance)
        distances_matrix.append(distances_row)

    return distances_matrix


# Calculate the volume of the simulation box
def get_simulation_volume(data_file):
    lx, ly, lz = 0, 0, 0

    with open(data_file, 'r') as file:
        lines = file.readlines()

        for line in lines:
            # Extract the box dimensions
            if "xlo xhi" in line:
                xlo, xhi = map(float, line.split()[:2])
                lx = xhi - xlo
            elif "ylo yhi" in line:
                ylo, yhi = map(float, line.split()[:2])
                ly = yhi - ylo
            elif "zlo zhi" in line:
                zlo, zhi = map(float, line.split()[:2])
                lz = zhi - zlo

    return lx * ly * lz


# get the time step of the simulation
def get_simulation_time_step(data_file):

    dt = 0.0
    with open(data_file, 'r') as file:
        lines = file.readlines()

        for line in lines:
            # Extract the timestep
            if line.startswith("timestep"):
                dt = float(line.split()[1])
        for line in lines:
            # Extract how frequently a frame is written into traj file
            if "dump mydump" in line:
                nframe = float(line.split()[4])

        dt = dt * nframe
        dt = dt * 1e-15

    return dt 


# Retrieve the cutoff range of Lock-and-Key potential
def get_LAK_cutoff(data_file):

    cutoff = 0
    with open(data_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Extract the LAK cotoff 
            if "lock/key" in line:
                cutoff = line.split()[-1]
    return float(cutoff)



def calculate_K_d_concentration(frames, volume, distance_threshold, time_per_frame):
    NA = 6.0221408e23               # Avogadro's number
    volume_liters = volume * 1e-27  # cubic Angstroms to liters
    nframes = len(frames)

    if volume == 0 or len(frames) == 0:
        print()
        logging.error("Error: Volume or number of frames is zero.\n")
        return None

    complex_count = 0
    free_count  = 0
    survival_times = []        # List to store survival times of each complex
    unbound_times  = []        # List to store unbound times of each complex
    event_times = []           # List to store all binding and unbinding events
    current_survival_time = 0  # Counter for the current survival time
    current_unbound_time  = 0  # Counter for the current unbound time
    event_in_progress = None   # Track the current event type

    for frame_positions in frames:
        distances_matrix = calculate_distances(frame_positions)
        distances = [dist for sublist in distances_matrix for dist in sublist]
        if any(d < distance_threshold for d in distances):
            if event_in_progress != "complex":
                if current_unbound_time > 0:
                    event_times.append(("free", current_unbound_time))
                    unbound_times.append(current_unbound_time)
                    current_unbound_time = 0
                event_in_progress = "complex"
            current_survival_time += 1
            complex_count += 1
        else:
            if event_in_progress != "free":
                if current_survival_time > 0:
                    event_times.append(("complex", current_survival_time))
                    survival_times.append(current_survival_time)
                    current_survival_time = 0
                event_in_progress = "free"
            current_unbound_time += 1
            free_count += 1

   # Add the last event if it wasn't added
    if current_survival_time > 0:
        event_times.append(("complex", current_survival_time))
        survival_times.append(current_survival_time)
    elif current_unbound_time > 0:
        event_times.append(("free", current_unbound_time))
        unbound_times.append(current_unbound_time)

    # List to store event information
    event_info = []

    # Collect event information
    for event_type, duration in event_times:
        event_description = f"{event_type}: {duration} frames ({1e6 * duration * time_per_frame:.4f} microseconds)"
        event_info.append(event_description)

    average_survival_time = sum(survival_times) / len(survival_times) if survival_times else 0
    average_unbound_time  = sum(unbound_times)  / len(unbound_times) if unbound_times else 0
    max_survival_time = max(survival_times) if survival_times else 0
    max_unbound_time  = max(unbound_times)  if unbound_times else 0


    # total time spent in each state
    time_bound   = average_survival_time * time_per_frame
    time_unbound = average_unbound_time  * time_per_frame

    print(f"\n      # of free frames:  {free_count} \n")
    print(f"      # of Complex frames: {complex_count} \n")
    print(f"      Average Complex Survival frames: {round(average_survival_time,2)} frames \n")
    print(f"      Maximum Complex Survival frames: {max_survival_time} frames \n")
    print(f"      Average Unbound frames: {round(average_unbound_time,2)} frames \n")
    print(f"      Maximum Unbound frames: {max_unbound_time} frames \n")

    
    moles_H3   = free_count  / NA
    moles_H4   = free_count  / NA
    moles_H3H4 = complex_count / NA

    concentration_H3   = moles_H3   / (volume_liters * nframes)
    concentration_H4   = moles_H4   / (volume_liters * nframes)
    concentration_H3H4 = moles_H3H4 / (volume_liters * nframes)

    print(f"     Concentration of H3 or H4:      {concentration_H3:.2e} M. \n")
    print(f"     Concentration of H3-H4 complex: {concentration_H3H4:.2e} M. \n")


    if time_bound == 0:
        K_off = float('inf')  
    else:
        K_off = 1 / time_bound

    if time_unbound == 0 or concentration_H3 == 0:
        K_on = float('inf') 
    else:
        K_on = 1 / (time_unbound * concentration_H3)

    print(f"     K_off: {K_off:.2e} s^-1\n") 
    print(f"     K_on : {K_on:.2e}  (s M)^-1\n")

    # Calculate K_d
    if concentration_H3H4 == 0:
        K_d = float('inf') 
    else:
        K_d = (concentration_H3 * concentration_H4) / concentration_H3H4

    K_d_off_on = K_off / K_on if K_on != 0 else float('inf')

    return K_d, K_d_off_on, event_info

#   ept ZeroDivisionError:
#        print()
#        logging.error("Error: Division by zero in concentration calculation.\n")
#        return None


def define_chunks(total_lines, lines_per_frame, num_cpus):
    chunk_size = total_lines // num_cpus
    chunks = []

    current_start = 0
    while current_start < total_lines:
        current_end = min(current_start + chunk_size, total_lines)
        # Adjust to frame boundary
        if current_end + lines_per_frame <= total_lines:
            current_end += lines_per_frame - (current_end % lines_per_frame)
        chunks.append((current_start, current_end))
        current_start = current_end

    return chunks


def get_total_lines(filename):
    result = subprocess.run(['wc', '-l', filename], stdout=subprocess.PIPE)
    return int(result.stdout.split()[0])


def extract_last_n_particles_from_chunk(xyz_file, n, start_line, end_line):
    """
    Extract the positions of the last n particles from a chunk of an XYZ trajectory.
    """
    frames = []
    with open(xyz_file, 'r') as file:
        for _ in range(start_line):
            file.readline()  # Skip lines until start of the chunk

        current_line = start_line
        while current_line < end_line:
            header = file.readline()
            current_line += 1
            if not header:
                break  # End of file or end of chunk
            num_atoms = int(header.strip())
            file.readline()  # Skip xyz comment line
            current_line += 1

            frame_positions = []
            for _ in range(num_atoms - n):
                file.readline()  # Skip xyz comment lines
                current_line += 1
            for _ in range(n):
                coords = list(map(float, file.readline().split()[1:4]))
                frame_positions.append(coords)
                current_line += 1
            frames.append(frame_positions)
    return frames


def process_chunk(args):
    xyz_file, n, start_line, end_line, volume, LAK_cutoff, dt = args
    frames = extract_last_n_particles_from_chunk(xyz_file, n, start_line, end_line)
    K_d, K_d_off_on, event_info = calculate_K_d_concentration(frames, volume, LAK_cutoff, dt)
    return start_line, end_line, (K_d, K_d_off_on, event_info)    


g = open("K_d_values.txt", 'a')

def main():
    xyz_file = "h3-h4.xyz"
    num_sites = 2
    num_particles = 2 
    volume = get_simulation_volume("hist.txt")
    dt = get_simulation_time_step("bd-in.lammps")
    LAK_cutoff = get_LAK_cutoff("interactions.in")
    num_cpus = 8 

    # get the total number of lines in the file
    total_lines = get_total_lines(xyz_file) 
    lines_per_frame = num_particles + 2 

    # Define chunks
    chunks = define_chunks(total_lines, lines_per_frame, num_cpus)

    # Parallel processing
    with Pool(processes=num_cpus) as pool:
        # Create tuples of arguments for each chunk
        chunk_args = [(xyz_file, num_sites, start, end, volume, LAK_cutoff, dt) for start, end in chunks]

        # Map each chunk to the process_chunk function
        results = pool.map(process_chunk, chunk_args)

    with open("K_d_values.txt", 'a') as g:
        for start_line, end_line, (K_d, K_d_off_on, event_info) in results:
            g.write(f"Chunk {start_line}-{end_line}:\n")
            g.write(f"K_d: {K_d:.4e}, K_d_off_on: {K_d_off_on:.4e}\n")
            for event in event_info:
                g.write(event + '\n')
            g.write('\n')  
 
    g.close()

if __name__ == "__main__":
    main()
