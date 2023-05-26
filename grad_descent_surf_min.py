import sys, os, pymol
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

atomic_radii = {
    "h" : 1.2, "c": 1.7,  "n": 1.55,  "o": 1.52, "f": 1.47,
    "p" : 1.8, "s": 1.8, "cl": 1.75, "br": 1.85, "i": 1.98,
    "mn":1.61, }

def load_protein_coordinates(file_path, molecule):
    pymol.cmd.load(file_path, "protein")
    cenOfMass = pymol.cmd.centerofmass()
    pymol.cmd.hide("everything", "protein")
    # Set the background color to white
    pymol.cmd.bg_color("white")
    pymol.cmd.show("surface", "protein")
    pymol.cmd.set("gaussian_resolution", 0.1)  # Set Gaussian surface resolution
    pymol.cmd.set("solvent_radius", 1.4)  # Set solvent radius
    pymol.cmd.save(molecule+"_gauss_surface.pse")  # Save PyMOL session as pse file

    pymol.cmd.map_new("protein_map", "gaussian", 1.0, "all", 0)

    pymol.cmd.isosurface("protein_surface", "protein_map")

    with open(file_path, "r") as file:
        lines = file.readlines()
        coordinates = []
        radii = []
        for line in lines:
            if line.startswith("ATOM"):
                parts = line.split()
                element = parts[-1]
                x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
                coordinates.append([x, y, z])
                if element.lower() not in atomic_radii.keys():
                    print(f" The {element} is not available in your atomic radii library! Update your library for {element}!")
                radii.append(atomic_radii.get(element.lower()))
    return np.array(coordinates), np.array(radii),  molecule+"_gauss_surface.pse", cenOfMass

def gradient_sphere_atom_distance_sq_diff(params, protein_coords, radii, solvent_radius = 1.4):
    x0, y0, z0, radius = params
    sphere_center = np.array([x0, y0, z0])

    distances = cdist([sphere_center], protein_coords).flatten()
    effective_radii = distances - radii - solvent_radius
    diffs = effective_radii - radius

    # Calculate gradient with respect to sphere center
    grad_center = np.sum(2 * diffs[:, np.newaxis] * (protein_coords - sphere_center), axis=0)

    # Calculate gradient with respect to radius
    grad_radius = -2 * np.sum(diffs)

    return np.concatenate((grad_center, [grad_radius]))

def sphere_atom_distance_sq_sum(params, protein_coords, radii, solvent_radius = 1.4):
    x0, y0, z0, radius = params
    sphere_center = np.array([x0, y0, z0])

    distances = cdist([sphere_center], protein_coords).flatten()
    effective_radii = distances - radii - solvent_radius
    diffs = effective_radii - radius
    sq_diffs = diffs ** 2

    return np.sum(sq_diffs)


# Load protein atomic coordinates and radii
file_path = sys.argv[1] # "/home/hesam/chromatin_project/1kx5_B.pdb"
molecule = sys.argv[2]
protein_coords, radii, pse_file, centerofmass= load_protein_coordinates(file_path, molecule)
avg_distance = np.mean(np.sqrt(np.sum((protein_coords - centerofmass) ** 2, axis=1)) + radii)
initial_guess = np.concatenate((centerofmass,[15]))

result = minimize(sphere_atom_distance_sq_sum, 
                  initial_guess,
                  args=(protein_coords, radii), 
                  #method = "SLSQP",
                  method = "L-BFGS-B",
                  jac=gradient_sphere_atom_distance_sq_diff,
                  options={'maxiter': 100000, 'gtol': 1e-6,'eps':1e-5, "disp":True},# 'maxfun': 100000, 'maxgrad': 100000},
                  bounds=[(centerofmass[0]-100, centerofmass[0]+100), (centerofmass[1]-100, centerofmass[1]+100), (centerofmass[2]-100, centerofmass[2]+100), (1, 200)]
       )

optimized_sphere = result.x

# Optimize sphere parameters
optimized_sphere    = optimized_sphere.reshape(1,4)
optimized_cenofmass = optimized_sphere[0,:3]
optimized_radius    = optimized_sphere[0,-1]
print("initial guess: ", initial_guess)
print("Optimized sphere center :", optimized_cenofmass)
print("Optimized radius        :", optimized_radius)


#Set the background color to white
pymol.cmd.bg_color("white")

# Add a sphere with the optimized center and radius
sphere_name = "optimized_sphere"
pymol.cmd.load(pse_file)
pymol.cmd.pseudoatom(sphere_name, pos=optimized_cenofmass.tolist(), vdw=optimized_radius)

# Set the sphere color to white and transparency to 0.5 (semi-transparent)
pymol.cmd.set("sphere_transparency", 0.5, sphere_name)
pymol.cmd.color("white", sphere_name)

# Add shading to the sphere (use depth cueing to add some shading effect)
pymol.cmd.set("depth_cue", 1)
pymol.cmd.set("fog_start", 0.4)

# Show the sphere
pymol.cmd.show("spheres", sphere_name)

# Increase the resolution of the surface and the sphere
pymol.cmd.set("surface_quality", 1)
pymol.cmd.set("sphere_quality", 2)

# Save the final PyMOL session
pymol.cmd.save(molecule+"_output.pse")

# Save a high-resolution image
pymol.cmd.ray(1920, 1080)
#pymol.cmd.png(molecule+"_output.png", dpi=300)

# Terminate PyMOL
pymol.cmd.quit()
