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

# Loop through the atoms and set colors and radii based on the periodic pattern
for {set i 0} {$i < $num_atoms} {incr i 25} {
    set red_start $i
    set red_end [expr {$i + 7}]
    set white_start [expr {$i + 9}]
    set white_end [expr {$i + 19}]
    set purple_start [expr {$i + 20}]
    set purple_end [expr {$i + 22}]
    set purple_end2 [expr {$i + 8}]
    set pink_start [expr {$i + 23}]
    set pink_end [expr {$i + 24}]
    
    # Ensure the indices are within the total number of atoms
    if {$red_end >= $num_atoms} { set red_end [expr {$num_atoms - 1}] }
    if {$white_end >= $num_atoms} { set white_end [expr {$num_atoms - 1}] }
    if {$pink_end >= $num_atoms} { set pink_end [expr {$num_atoms - 1}] }
    if {$purple_end >= $num_atoms} { set purple_end [expr {$num_atoms - 1}] }
    if {$purple_end2 >= $num_atoms} { set purple_end2 [expr {$num_atoms - 1}] }
    
    # Set radius for red atoms
    set hist [atomselect top "index $red_start to $red_end"]
    $hist set radius 16.5

    # Representation for red atoms
    mol representation VDW $sphere_scale $sphere_resolution
    mol selection "index $red_start to $red_end"
    mol addrep 0
    mol modcolor [expr {[molinfo top get numreps] - 1}] 0 ColorID 1  ;# 1 for red
    
    # Set radius for white atoms
    set dna [atomselect top "index $white_start to $white_end"]
    #$dna set radius 14.5
    $dna set radius 15.5

    # Representation for white atoms
    mol representation VDW $sphere_scale $sphere_resolution
    mol selection "index $white_start to $white_end"
    mol addrep 0
    mol modcolor [expr {[molinfo top get numreps] - 1}] 0 ColorID 8  ;# 0 for white

    # Set radius for purple atoms
    set linker_dna [atomselect top "index $purple_end2 $purple_start to $purple_end"]
    #$dna set radius 14.5
    $linker_dna set radius 15.5

    # Representation for purple atoms
    mol representation VDW $sphere_scale $sphere_resolution
    mol selection "index $purple_end2 $purple_start to $purple_end"
    mol addrep 0
    mol modcolor [expr {[molinfo top get numreps] - 1}] 0 ColorID 11  ;# 11 for purple

    # Set radius for pink atoms
    set tail [atomselect top "index $pink_start to $pink_end"]
    $tail set radius 10

    # Representation for pink atoms
    mol representation VDW $sphere_scale $sphere_resolution
    mol selection "index $pink_start to $pink_end"
    mol addrep 0
    mol modcolor [expr {[molinfo top get numreps] - 1}] 0 ColorID 9  ;# 9 for pink
}

# Center and display the molecule
for {set i 0} {$i < [molinfo top get numreps]} {incr i} {
    mol modstyle $i 0 VDW $sphere_scale $sphere_resolution
}
