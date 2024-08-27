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

class Angle:
    def __init__(self, angle_index, angle_type, atom_index_1, atom_index_2, atom_index_3, comment): 
        self.angle_index  = angle_index
        self.angle_type   = angle_type
        self.atom_index_1 = atom_index_1
        self.atom_index_2 = atom_index_2
        self.atom_index_3 = atom_index_3
        self.comment      = comment


def generate_random_displacement(min_distance=13, max_distance=16):

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

    f = open(f"{n_particles}-octamer.txt", 'w')

    atoms  = {}
    bonds  = []
    angles = []
    atom_index  = 1
    bond_index  = 1
    angle_index = 1
    max_attempts = 1000
    parent_atom_types = [1, 2, 3, 4] 
    parent_sites = {}
    interaction_groups = {
        ('1', '2'):  1, ('2', '1'):  2, 
        ('1', '3'):  3, ('3', '1'):  4, 
        ('2', '3'):  5, ('3', '2'):  6, 
        ('2', '4'):  7, ('4', '2'):  8,
        ('1', '4'):  9, ('4', '1'): 10,
        ('3', '4'): 11, ('4', '3'): 12, 
        ('1', '1'): 13, ('2', '2'): 14, 
        ('3', '3'): 15, ('4', '4'): 16, 
    }

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
        ('12', '4', '16'): 12,
    }

    for i in range(1, n_particles + 1):
        attempts = 0
        while True:
            parent_position = np.random.uniform(10, 970, (3))
            if not check_overlap(parent_position, atoms, 20 ) or attempts > max_attempts:
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
                site_position = parent_position + generate_random_displacement()
                site_comment = f"{parent_atom_type} - {site}"
                site_atom = Atom(atom_index, i, site_atom_type, *site_position, site_comment, is_site=True)
                atoms[atom_index] = site_atom
                bond = Bond(bond_index, bond_type, parent_index, atom_index, site_comment)
                bonds.append(bond)

                # Store site indices by parent
                if parent_index not in parent_sites:
                    parent_sites[parent_index] = []
                parent_sites[parent_index].append(atom_index)

            atom_index += 1
            bond_index += 1

        #atom_index += 1

    # Generate angles from site lists
    for parent_index, site_indices in parent_sites.items():
        #print(f"Generating angles for parent {parent_index} with sites {site_indices}")
        parent_type = str(atoms[parent_index].atom_type)
    
        for j in range(len(site_indices)):
            for k in range(j+1 , len(site_indices)):
                type1 = str(atoms[site_indices[j]].atom_type)
                type2 = str(atoms[site_indices[k]].atom_type)
                angle_key1 = (type1, parent_type, type2)
                angle_key2 = (type2, parent_type, type1)
                #print(angle_key1)
                #print(angle_key2)
    
                #print(f"Checking angle keys: {angle_key1} and {angle_key2}")
                # Check if either angle configuration is valid
                if angle_key1 in angle_types:
                    #print(f"Match found for key {angle_key1}")
                    angle_type = angle_types[angle_key1]
                    angle_comment = f"{type1}-{parent_type}-{type2}"
                    angle = Angle(angle_index, angle_type, site_indices[j], parent_index, site_indices[k], angle_comment)
                    angles.append(angle)
                    angle_index += 1
                elif angle_key2 in angle_types:
                    #print(f"Match found for key {angle_key2}")
                    angle_type = angle_types[angle_key2]
                    angle_comment = f"{type2}-{parent_type}-{type1}"
                    angle = Angle(angle_index, angle_type, site_indices[k], parent_index, site_indices[j], angle_comment)
                    angles.append(angle)
                    angle_index += 1


    data = [] 
    for key, atom in atoms.items():
        data.append(atom.coordinates)

    data = np.array(data)

    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
    z_min, z_max = np.min(data[:, 2]), np.max(data[:, 2])
    padding = 100

    f.write("# LAMMPS data file\n\n")

    f.write(f"   {str(n_particles*4)}   atoms\n")
    f.write(f"   {str(len(bonds))}   bonds\n")
    f.write(f"   {str(len(angles))}   angles\n\n")

    f.write("   16    atom types \n")
    f.write("   12    bond types \n")
    f.write("   12    angle types\n\n")

    f.write(f"   {str(round(x_min - padding, 2))}  {str(round(x_max + padding, 2))} xlo xhi\n")
    f.write(f"   {str(round(y_min - padding, 2))}  {str(round(y_max + padding, 2))} ylo yhi\n")
    f.write(f"   {str(round(z_min - padding, 2))}  {str(round(z_max + padding, 2))} zlo zhi\n\n")

    f.write("Masses\n\n")
    f.write("   1    15273\n")
    f.write("   2    11390\n")
    f.write("   3    13789\n")
    f.write("   4    13990\n")
    f.write("   5     100\n")
    f.write("   6     100\n")
    f.write("   7     100\n")
    f.write("   8     100\n")
    f.write("   9     100\n")
    f.write("   10    100\n")
    f.write("   11    100\n")
    f.write("   12    100\n")
    f.write("   13    100\n")
    f.write("   14    100\n")
    f.write("   15    100\n")
    f.write("   16    100\n")


    f.write("\nAtoms\n\n")
    for at in atoms:
        atom = atoms[at]
        f.write(f"   {atom.atom_index:<6}{atom.mol_id:<7}{atom.atom_type:<6}{atom.coordinates[0]:<13.6f}{atom.coordinates[1]:<13.6f}{atom.coordinates[2]:<13.6f}  # {atom.comment}\n")

    f.write("\nBonds\n\n")
    for b in bonds:
        f.write(f"   {b.bond_index:<7}{b.bond_type:<8}{b.atom_index_1:<8}{b.atom_index_2:<8} # {b.comment}\n")

    f.write("\nAngles\n\n")
    for a in angles:
        f.write(f"   {a.angle_index:<7}{a.angle_type:<8}{a.atom_index_1:<8}{a.atom_index_2:<8}{a.atom_index_3:<8}# {a.comment}\n")

    return atoms, bonds, angles


# Generate a system of n particles with associated site particles
n_particles = 256 
atoms, bonds, angles = generate_system(n_particles)
