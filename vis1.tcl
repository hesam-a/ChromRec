# Check if the file name is provided as an argument
if { $argc != 1 } {
    puts "Usage: vmd -e script.tcl -args filename.xyz"
    exit
}

# Get the filename from the arguments
set filename [lindex $argv 0]

# Load the XYZ file
mol new $filename type xyz waitfor all

# Delete the default representation
mol delrep 0 0

# Set sphere resolution and scale
set sphere_resolution 70
set sphere_scale 1.0

# Get the total number of atoms
set num_atoms [molinfo top get numatoms]

# Select all atoms
set all_atoms [atomselect top "all"]

# Get atomic numbers (chemical symbols in this case)
set atomic_numbers [$all_atoms get name]

# Initialize variables
set in_white_block 0
set white_counter 0

# Loop through the atoms and set colors and radii based on the periodic pattern
for {set i 0} {$i < $num_atoms} {incr i} {
    set atomic_number [lindex $atomic_numbers $i]

    if {$atomic_number == "H" || $atomic_number == "He" || $atomic_number == "Li" || $atomic_number == "Be"} {
        # Set radius for red atoms
        set hist [atomselect top "index $i"]
        $hist set radius 16.5

        # Representation for red atoms
        mol representation VDW $sphere_scale $sphere_resolution
        mol selection "index $i"
        mol addrep 0
        mol modcolor [expr {[molinfo top get numreps] - 1}] 0 ColorID 1  ;# 1 for red

        # Reset white block tracking
        set in_white_block 0
        set white_counter 0

    } elseif {$atomic_number == "B"} {
        if {$in_white_block == 0} {
            # Start a new white block
            set in_white_block 1
            set white_counter 0

            # First particle of the block should be purple
            set purple_atom [atomselect top "index $i"]
            $purple_atom set radius 15.5

            # Representation for purple atom
            mol representation VDW $sphere_scale $sphere_resolution
            mol selection "index $i"
            mol addrep 0
            mol modcolor [expr {[molinfo top get numreps] - 1}] 0 ColorID 11  ;# 11 for purple

        } elseif {$white_counter < 11} {
            # Continue the white block for the next 12 particles
            set idx [expr {$i}]
            if {$idx >= $num_atoms} { break }
            set white_atom [atomselect top "index $idx"]
            $white_atom set radius 15.5

            # Representation for white atoms
            mol representation VDW $sphere_scale $sphere_resolution
            mol selection "index $idx"
            mol addrep 0
            mol modcolor [expr {[molinfo top get numreps] - 1}] 0 ColorID 8  ;# 8 for white

            incr white_counter

        } else {
            # Switch to purple after the white block
            set purple_atom [atomselect top "index $i"]
            $purple_atom set radius 15.5

            # Representation for purple atoms
            mol representation VDW $sphere_scale $sphere_resolution
            mol selection "index $i"
            mol addrep 0
            mol modcolor [expr {[molinfo top get numreps] - 1}] 0 ColorID 11  ;# 11 for purple
        }

    } else {
        if {$white_counter >= 11} {
            # Continue purple until the atomic number changes to 1, 2, 3, or 4
            set purple_atom [atomselect top "index $i"]
            $purple_atom set radius 15.5

            # Representation for purple atoms
            mol representation VDW $sphere_scale $sphere_resolution
            mol selection "index $i"
            mol addrep 0
            mol modcolor [expr {[molinfo top get numreps] - 1}] 0 ColorID 11  ;# 11 for purple
        }
    }
}

# Center and display the molecule
for {set i 0} {$i < [molinfo top get numreps]} {incr i} {
    mol modstyle $i 0 VDW $sphere_scale $sphere_resolution
}
