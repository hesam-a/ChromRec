import numpy as np
import matplotlib.pyplot as plt

# Load distances file
distances = []
pairs = []

with open("distances.txt", "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        pair_index = int(parts[0])
        nuc1, nuc2 = int(parts[1]), int(parts[2])
        pairs.append((nuc1, nuc2))
        distances.append([float(d) for d in parts[3:]])

# Convert to NumPy array for easy processing
distances = np.array(distances)

# Plot histogram of distances
plt.figure(figsize=(8, 6))
for i, (nuc1, nuc2) in enumerate(pairs):
    plt.hist(distances[i].flatten(), label=f"Pair {nuc1}-{nuc2}", bins=50, density=True, alpha=0.75)
plt.xlabel("Distance (Å)")
plt.ylabel("Probability Density")
plt.title("Histogram of Nucleosome Distances")
plt.legend()
plt.savefig("distance_histogram.png")

# Load autocorrelation file
autocorr_data = []
pairs_ac = []

with open("autocorrelation.txt", "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        pair_index = int(parts[0])
        nuc1, nuc2 = int(parts[1]), int(parts[2])
        pairs_ac.append((nuc1, nuc2))
        autocorr_data.append([float(a) for a in parts[3:]])

# Convert to NumPy array
autocorr_data = np.array(autocorr_data)

# Plot autocorrelation decay for each pair
plt.figure(figsize=(8, 6))
for i, (nuc1, nuc2) in enumerate(pairs_ac):
    plt.plot(autocorr_data[i], label=f"Pair {nuc1}-{nuc2}")
plt.xlabel("Time Lag (frames)")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation Decay for Nucleosome Pairs")
plt.legend()
plt.savefig("autocorrelation_decay.png")

plt.show()
