import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import glob, subprocess, time, os, gc, sys

angle_types = {
    (5, 1, 7)  : (1,  "121" , "H3-H3-H2A"), 
    (5, 1, 13) : (2,  "136" , "H3-H4-H2B"),
    (7, 1, 13) : (3,  "17"  , "H2A-H3-H2B"),
    (6, 2, 9)  : (4,  "46" , "H4-H3-H2A"),  
    (6, 2, 11) : (5,  "36" , "H4-H3-H2B"),  
    (9, 2, 11) : (6,  "13.5", "H2A-H4-H2B"),
    (8, 3, 10) : (7,  "12.8", "H3-H2A-H4"),
    (8, 3, 15) : (8,  "91"  , "H3-H2A-H2B"),
    (10, 3, 15): (9,  "80"  , "H4-H2A-H2B"),
    (14, 4, 12): (10, "10"  , "H3-H2B-H4"), 
    (14, 4, 16): (11, "75"  , "H3-H2B-H2A"),
    (12, 4, 16): (12, "83.5", "H4-H2B-H2A")
} 

def delete_files(files_for_deletion):
    for file in files_for_deletion:
        try:
            os.remove(file)
        except OSError as e:
            print(f"Error deleting file {file}: {e.strerror}")

def concatenate_files(angle_id, files_for_angle, output_filename):
    if files_for_angle:
        with open(output_filename, 'w') as outfile:
            subprocess.run(['cat'] + files_for_angle, stdout=outfile)

def read_and_plot_angles(angle_id, concatenated_filename, degree, angle_name, current_dir):
    if not os.path.exists(concatenated_filename):
        print(f"File {concatenated_filename} does not exist. Skipping.")
        return None

    plt.figure(figsize=(10, 6), dpi=100)
    angles = np.loadtxt(concatenated_filename)

    # Calculate mean and standard deviation
    mean_angle = np.mean(angles)
    std_dev_angle = np.std(angles)

    plt.hist(angles, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.title(f"Angle Distribution for Type {angle_id}, {angle_name}", fontweight='bold', fontsize=16)
    plt.xlabel("Angle (degrees)", fontweight='bold', fontsize=14, labelpad=14)
    plt.ylabel("Frequency", fontweight='bold', fontsize=14, labelpad=14)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Annotate mean and standard deviation on the plot
    textstr = f'Original angle: {degree}\nMean: {mean_angle:.2f}\nStd Dev: {std_dev_angle:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # Place text box in upper right in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(f"{current_dir}/angle_hist_{angle_id}_{angle_name}.png")
    plt.close()

    del angles
    gc.collect()

def plot_all_angle_types(angle_types, current_dir):
    all_files = glob.glob("chunk_*_angle_*_angles")
    
    for angle_triplet, angle_info in angle_types.items():
        angle_id, degree, angle_name = angle_info
        output_filename = f"angle_{angle_id}_angles"
        pattern = f"chunk_*_angle_{angle_id}_angles"
        files_for_angle = [filename for filename in all_files if filename.endswith(pattern[7:])]
        files_for_angle.sort(key=lambda x: int(x.split('_')[1]))

        concatenate_files(angle_id, files_for_angle, output_filename)
        delete_files(files_for_angle)
        read_and_plot_angles(angle_id, output_filename, degree, angle_name, current_dir)

start = time.time()

current_dir = sys.argv[1]
plot_all_angle_types(angle_types, current_dir)

end = time.time()
print(f"Time taken: {(end - start):.2f} seconds")
