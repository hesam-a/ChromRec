# Molecular weights of each amino acid (on H3 tail) in Da
amino_acid_weights = {
    'A': 89.09,
    'R': 174.20,
    'N': 132.12,
    'D': 133.10,
    'C': 121.16,
    'E': 147.13,
    'Q': 146.15,
    'G': 75.07,
    'H': 155.15,
    'I': 131.18,
    'L': 131.18,
    'K': 146.19,
    'M': 149.21,
    'F': 165.19,
    'P': 115.13,
    'S': 105.09,
    'T': 119.12,
    'W': 204.23,
    'Y': 181.19,
    'V': 117.15
}

# The first 38 residues of the histone H3 tail from the provided sequence
sequence_tail = "ARTKQTARKSTGGKAPRKQLATKAARKSAPATGGV"

# Calculate the total molecular weight
total_weight_tail = sum(amino_acid_weights[aa] for aa in sequence_tail)
print(f"\n total weight of the H3 tail:  {total_weight_tail:.2f}\n")
