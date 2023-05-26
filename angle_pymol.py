from pymol import cmd, stored
import prody
import numpy as np

# Load the PDB and DCD files
pdb_file = 'pan_dna.pdb'
dcd_file = 'dna.dcd'
cmd.load(pdb_file)
cmd.load_traj(dcd_file, state=0)  # Load trajectory into state 0 (all states)

file_list = ["res_0_idx.txt", "res_15_idx.txt", "res_30_idx.txt"]

selections = ["", "", ""]

for i, fil in enumerate(file_list):

    # Read the atom indices from the file
    with open(fil, "r") as file:
        atom_indices = file.read().strip().split()
    
    # Convert atom indices to integers
    atom_indices = [int(index) for index in atom_indices]
    
    # Select the atoms in PyMOL using the indices
    selection_name = "sel" + str(i+1)
    cmd.select(selection_name, f"index {'+'.join(map(str, atom_indices))}")
    selections[i] = selection_name  # store the selection names

    # Print indices for sel1
#    if selection_name == "sel1":
#        print(f"Indices for sel1: {atom_indices}")

nframes = cmd.count_states('all')  # Number of frames in the trajectory

print("Selections done.")

angles_list = []

# Loop over all frames
for i in range(nframes):
#    print(f"Processing frame {i+1}/{nframes}")
    cmd.frame(i+1)  # Set to frame i (PyMOL is 1-indexed)
    
    # Calculate center of mass for each selection
    model1 = cmd.get_model(selections[0], state=i+1).get_coord_list()
    model2 = cmd.get_model(selections[1], state=i+1).get_coord_list()
    model3 = cmd.get_model(selections[2], state=i+1).get_coord_list()

    com1 = prody.calcCenter(np.array(model1))
    com2 = prody.calcCenter(np.array(model2))
    com3 = prody.calcCenter(np.array(model3))
    
    # Create vectors and calculate angle
    vec1 = com2 - com1
    vec2 = com2 - com3
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(cos_angle)

    # Convert angle to degrees
    angle_degrees = np.degrees(angle)
#    print(angle_degrees)

#    angles_list.append(angle)
    angles_list.append(angle_degrees)

print("Angle calculations done.")

# Save angles to file
np.savetxt('angles_new_deg.txt', np.array(angles_list), fmt='%.5f')

pymol.cmd.quit()
