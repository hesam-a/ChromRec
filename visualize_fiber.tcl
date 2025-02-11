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

# Set sphere resolution and scale for VDW representations
set sphere_resolution 70
set sphere_scale 1.0

# Set a custom gray color for background using a predefined color ID (e.g., Color ID 0 for black)
color change rgb 0 0.902 0.902 0.902  ;# Set Color ID 0 to light gray (RGB values for grayscale 230)

# Now set the background to Color ID 0 (which is now light gray)
color Display Background 0

# Set custom colors for DNA and histones based on grayscale values from the EMT images

# For DNA (light gray, grayscale 213)
color change rgb 8 0.835 0.835 0.835 ; # Color ID 8 set to light gray for DNA

# For Histones (dark gray, grayscale 30)
color change rgb 2 0.118 0.118 0.118  ;# Color ID 2 set to dark gray for histones

# Representation for atomic numbers (assign radii for different components)
set h3 [atomselect top "atomicnumber 1"]
$h3 set radius 17.35

set h4 [atomselect top "atomicnumber 2"]
$h4 set radius 16.24

set h2A [atomselect top "atomicnumber 3"]
$h2A set radius 16.89

set h2B [atomselect top "atomicnumber 4"]
$h2B set radius 16.7

set dna [atomselect top "atomicnumber 5"]
$dna set radius 14.36

set h3tail [atomselect top "atomicnumber 6"]
$h3tail set radius 10.0

set h4tail [atomselect top "atomicnumber 7"]
$h4tail set radius 10.0

set acidpatch [atomselect top "atomicnumber 8"]
$acidpatch set radius 10.0

set h3tailac [atomselect top "atomicnumber 9"]
$h3tailac set radius 10.0

set h4tailac [atomselect top "atomicnumber 10"]
$h4tailac set radius 10.0

# Add representations and apply custom colors

# For histones (atomicnumber 1, 2, 3, 4), use dark gray (color ID 2)
mol selection "atomicnumber 1 2 3 4"
mol representation VDW $sphere_scale $sphere_resolution
mol material Opaque; #Transparent
mol addrep 0
mol modcolor [expr {[molinfo top get numreps] - 1}] 0 ColorID 2  ;# Apply dark gray to histones

# For DNA (atomicnumber 5), use light gray (color ID 8)
mol selection "atomicnumber 5"
mol representation VDW $sphere_scale $sphere_resolution
mol material Opaque; #Transparent
mol addrep 0
mol modcolor [expr {[molinfo top get numreps] - 1}] 0 ColorID 8  ;# Apply light gray to DNA

# For h3tail (atomicnumber 6), use silver (color ID 6)
mol selection "atomicnumber 6"
mol representation VDW $sphere_scale $sphere_resolution
mol material Opaque; #Transparent
mol addrep 0
mol modcolor [expr {[molinfo top get numreps] - 1}] 0 ColorID 6  ;# Apply light gray to DNA

# For h4tail (atomicnumber 7), use silver (color ID 6)
mol selection "atomicnumber 7"
mol representation VDW $sphere_scale $sphere_resolution
mol addrep 0
mol modcolor [expr {[molinfo top get numreps] - 1}] 0 ColorID 6  ;# Apply light gray to DNA

# For acidpatch (atomicnumber 8), use silver (color ID 6)
mol selection "atomicnumber 8"
mol representation VDW $sphere_scale $sphere_resolution
mol addrep 0
mol modcolor [expr {[molinfo top get numreps] - 1}] 0 ColorID 6  ;# Apply light gray to DNA

# For h3tailac (atomicnumber 10), use green (color ID 7)
mol selection "atomicnumber 9"
mol representation VDW $sphere_scale $sphere_resolution
mol material Opaque; #Transparent

mol addrep 0
mol modcolor [expr {[molinfo top get numreps] - 1}] 0 ColorID 7  ;# Apply light gray to DNA

# For h4tailac (atomicnumber 10), use green (color ID 7)
mol selection "atomicnumber 10"
mol representation VDW $sphere_scale $sphere_resolution
mol addrep 0
mol modcolor [expr {[molinfo top get numreps] - 1}] 0 ColorID 7  ;# Apply light gray to DNA

# Automatically adjust the view
display resetview

# Play the trajectory if it's available
animate goto start

# Set the view (e.g., align with y-axis)
rotate y by 90  ;# Align the view as needed

# Save the image to a PNG file
#render snapshot my_image.png
