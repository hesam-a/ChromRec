import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from itertools import combinations_with_replacement
import glob, subprocess, time, os, gc, sys

# Define your proteins
proteins = ["H3", "H4", "H2A", "H2B"]

def delete_files(files_for_deletion):
    for file in files_for_deletion:
        try:
            os.remove(file)
        except OSError as e:
            print(f"Error deleting file {file}: {e.strerror}")

def concatenate_files(interaction, files_for_interaction, output_filename):
    if files_for_interaction:  # Check if there are files to concatenate
        with open(output_filename, 'w') as outfile:
            subprocess.run(['cat'] + files_for_interaction, stdout=outfile)

def read_and_plot_distances(interaction, concatenated_filename, current_dir):
    if not os.path.exists(concatenated_filename):
        print(f"File {concatenated_filename} does not exist. Skipping.")
        return None

    plt.figure(figsize=(10, 6), dpi=100)
    distances = np.loadtxt(concatenated_filename)

    plt.hist(distances, bins=50, alpha=0.75, color='red', edgecolor='black')
    plt.title(f"Distances for {interaction}", fontweight='bold', fontsize=16)
    plt.xlabel("Distance (Å)", fontweight='bold', fontsize=14, labelpad=14)
    plt.ylabel("Frequency", fontweight='bold', fontsize=14, labelpad=14)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    mean = np.mean(distances)
    std_dev = np.std(distances)
    textstr = f'Mean: {mean:.2f}\nStd Dev: {std_dev:.2f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(f"{current_dir}/{interaction}_histogram.png")
    plt.close()

    del distances
    gc.collect()
    
    return mean

def plot_all_interactions_separately(proteins, current_dir):
    interaction_pairs = list(combinations_with_replacement(proteins, 2))
    all_files = glob.glob("chunk_*_*_*_distances")
    means = {}

    for pair in interaction_pairs:
        interaction = f"{pair[0]}-{pair[1]}"
        output_filename = f"{interaction}_distances"
        pattern1 = f"chunk_*_{pair[0]}_{pair[1]}_distances"
        pattern2 = f"chunk_*_{pair[1]}_{pair[0]}_distances"
        files_for_pair = [filename for filename in all_files if filename.endswith(pattern1[7:]) or filename.endswith(pattern2[7:])]
        files_for_pair.sort(key=lambda x: int(x.split('_')[1]))

        concatenate_files(interaction, files_for_pair, output_filename)
        delete_files(files_for_pair)
        
        mean = read_and_plot_distances(interaction, output_filename, current_dir)
        if mean is not None:
            means[interaction] = mean

    with open("means.txt", 'w') as f:
        for interaction, mean in means.items():
            f.write(f"{interaction}:{mean}\n")

current_dir = sys.argv[1]
start = time.time()
plot_all_interactions_separately(proteins, current_dir)
end = time.time()
print(f"Time taken: {(end - start):.2f} seconds")

