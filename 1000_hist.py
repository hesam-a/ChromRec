import numpy as np

class Atom:
    def __init__(self, atom_index, mol_id, atom_type, x, y, z, comment, is_site=False):
        self.atom_index = atom_index
        self.mol_id = mol_id
        self.atom_type = atom_type
        self.coordinates = np.array([x, y, z])
        self.comment = comment
        self.is_site = is_site
        self.bonds = []  # Will store Bond objects

class Bond:
    def __init__(self, bond_index, bond_type, atom_index_1, atom_index_2, comment):
        self.bond_index = bond_index
        self.bond_type = bond_type
        self.atom_index_1 = atom_index_1
        self.atom_index_2 = atom_index_2
        self.comment = comment

def generate_random_displacement(min_distance=10, max_distance=12):

    if np.random.rand() > 0.5:
        magnitude = np.random.uniform(min_distance, max_distance)
    else:
        magnitude = -np.random.uniform(min_distance, max_distance)

    theta = np.random.uniform(0, np.pi*2)
    phi = np.random.uniform(0, np.pi)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    displacement = np.array([x, y, z]) * magnitude

    return displacement


# Function to check if a new position overlaps with existing atoms
def check_overlap(new_position, existing_atoms, threshold):
    for atom in existing_atoms.values():
        if np.linalg.norm(atom.coordinates - new_position) < threshold:  # Assuming a minimum distance to avoid overlap
            return True
    return False

def generate_system(n_particles):
    atoms = {}
    bonds = []
    atom_index = 1
    bond_index = 1
    max_attempts = 1000
    parent_atom_types = [1, 2, 3, 4] 
    interaction_groups = {
        ('1', '2'):  1, ('2', '1'):  2, 
        ('1', '3'):  3, ('3', '1'):  4, 
        ('2', '3'):  5, ('3', '2'):  6, 
        ('2', '4'):  7, ('4', '2'):  8,
        ('1', '4'):  9, ('4', '1'): 10,
        ('3', '4'): 11, ('4', '3'): 12, 
    }

    for i in range(1, n_particles + 1):
        attempts = 0
        while True:
            parent_position = np.random.uniform(-200, 200, (3))
            if not check_overlap(parent_position, atoms,10 ) or attempts > max_attempts:
                break
            attempts += 1
        
        if attempts > max_attempts:
            print(f"Could not place atom {atom_index} without overlap after {max_attempts} attempts even with 5 A threshold.")
            continue

        # Assign atom type to parent particle
        parent_atom_type = np.random.choice(parent_atom_types)
        #print(f"atom type: {parent_atom_type}")
        parent_comment = f"{parent_atom_type}"
        parent_atom = Atom(atom_index, i, parent_atom_type, *parent_position, parent_comment, is_site=False)
        coords_str = "   ".join([f"{coord:.4f}" for coord in parent_atom.coordinates])
        #print(f" {parent_atom.atom_index}   {parent_atom.mol_id}   {parent_atom.atom_type}   {coords_str}  # {parent_atom.comment}")
        atoms[atom_index] = parent_atom
        parent_index = atom_index
        atom_index += 1

        # Generate site particles for this parent
        for site in range(1,5):
            if site == parent_atom_type:
                continue

            site_str = str(site)
            parent_atom_type_str = str(parent_atom_type)

            interaction_key = (parent_atom_type_str, site_str)
     
            if interaction_key in interaction_groups:
                bond_type = interaction_groups[interaction_key] 
                site_atom_type = interaction_groups[interaction_key] + 4
                #print(f"Interaction between type {parent_atom_type_str} and {site_str} has value: {site_atom_type}")     

            site_position = parent_position + generate_random_displacement()
            site_comment = f"{parent_atom_type} - {site}"
            site_atom = Atom(atom_index, i, site_atom_type, *site_position, site_comment, is_site=True)
            #coords_str = "   ".join(map(str, site_atom.coordinates))
            #coords_str = "   ".join([f"{coord:.4f}" for coord in site_atom.coordinates])
            #print(f" {site_atom.atom_index}   {parent_atom.mol_id}   {site_atom.atom_type}   {coords_stri}  # {site_atom.comment}")
            atoms[atom_index] = site_atom

            # Create bond between parent and this site particle
            bond = Bond(bond_index, bond_type, parent_index, atom_index, site_comment) 
            bonds.append(bond)

            atom_index += 1
            bond_index += 1

    print("\nAtoms\n")

    for at in atoms:
        atom = atoms[at]
        print(f" {atom.atom_index:<6}{atom.mol_id:<7}{atom.atom_type:<6}{atom.coordinates[0]:<13.6f}{atom.coordinates[1]:<13.6f}{atom.coordinates[2]:<13.6f}  # {atom.comment}")

    print("\nBonds\n")

    for b in bonds:
        print(f" {b.bond_index:<7}{b.bond_type:<8}{b.atom_index_1:<8}{b.atom_index_2:<8} # {b.comment}")
    return atoms, bonds

# Generate a system of n particles with associated site particles
n_particles = 500 
atoms, bonds = generate_system(n_particles)
