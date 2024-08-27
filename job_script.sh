#!/bin/bash 

start=$(date +%s)

# Initial parameters
total_duration=300  # Adjust as needed
step=10
base_restart_value=1000000000
template_file="bd-in.template.lammps"
base_seed=12345  # Define a base seed for the random number generator

# Loop through the time intervals
for (( current_start=0; current_start<total_duration; current_start+=step )); do
    current_end=$((current_start + step))
    new_seed=$((base_seed + current_start/step))  # Increment the seed for each run

    echo " "
    echo " current step "
    echo $current_end
    #echo $new_seed
    echo " "

    mkdir -p "$current_end"

    lmp -in bd-in.lammps

    #rm *_angles*
    #python calculate_angle.py octamer.xyz
    #python angle_plot.py "$current_end"
    
    rm *_dist*
    python dist_22.py core.xyz 
    python dist_plott.py "$current_end"

    sed -i "s/brownian 300.0 12345/brownian 300.0 $new_seed/" bd-in.lammps

    # Calculate the new restart file value
    restart_value=$((base_restart_value + current_start/step * base_restart_value))
    echo $restart_value

    # Ensure current_end is less than total_duration before copying the template file
    if [ "$current_end" -lt "$total_duration" ]; then
        # Copy the template file and modify the LAMMPS input file
        cp "$template_file" bd-in.lammps
    fi

    # Update the restart file value and the random seed in the LAMMPS input file
    sed -i "s/read_restart restart_octamer.lammps.1000000000/read_restart restart_octamer.lammps.$restart_value/" bd-in.lammps
done

end=$(date +%s)
runtime=$((end - start))

echo -e "\nTotal runtime: $runtime seconds\n"
