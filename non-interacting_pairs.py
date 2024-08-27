# Step 1: Open and read the "interactions.in" file
interactions_file_path = "interactions.in"
interactions_found = set()

with open(interactions_file_path, 'r') as file:
    for line in file:
        if line.startswith("pair_coeff"):
            _, type1, type2, *_ = line.split()
            interactions_found.add(f"{type1} {type2}")
            interactions_found.add(f"{type2} {type1}")  # Add reverse since interactions are bidirectional

# Step 2: Identify all unique atom types
unique_atom_types = {int(type_) for pair in interactions_found for type_ in pair.split()}

# Step 3: Generate the complete set of possible pair interactions
all_possible_interactions = {f"{i} {j}" for i in unique_atom_types for j in unique_atom_types if i <= j}

# Step 4: Identify missing interactions
missing_interactions = all_possible_interactions - interactions_found

# Step 5: Print out the missing interactions in the specified format
for interaction in sorted(missing_interactions, key=lambda x: (int(x.split()[0]), int(x.split()[1]))):
    type1, type2 = interaction.split()
    print(f"pair_coeff {type1} {type2} none")

