# Load the PDB file
mol new only_nucl_init.pdb
set mol_id [molinfo top]

# Identify lysine and arginine residues
set lysines [atomselect top "resname LYS"]
set arginines [atomselect top "resname ARG"]

# Identify the DNA
set dna [atomselect top "resname ADE GUA CYT THY"]

# Get the list of residues for lysines and arginines
set lysine_residues [lsort -unique [$lysines get residue]]
set arginine_residues [lsort -unique [$arginines get residue]]

#set num_lysines [llength [lsort -unique $lysine_residues]]
#puts "Number of lysine residues: $num_lysines"
#
#set num_arginines [llength [lsort -unique $arginine_residues]]
#puts "Number of arginine residues: $num_arginines"

# Initialize lists to store points
set lys_arg_points [list]
set dna_points [list]

# Open the file for writing
set output_file [open "contact_residues.txt" "w"]
set processed_dna_residues {}

# Loop over all the lysine residues
foreach residue $lysine_residues {
    set current_residue [atomselect top "resname LYS and residue $residue"]
    set contacts [measure contacts 5 $current_residue $dna]
    if {[llength [lindex $contacts 0]] > 0 || [llength [lindex $contacts 1]] > 0} {
        puts "Lysine residue $residue is in contact with DNA"
        set center_of_mass [measure center $current_residue weight mass]
        puts $output_file "Lysine, $residue, $center_of_mass"
        foreach contact_residue [lindex $contacts 0] contact_dna [lindex $contacts 1] {
            lappend lys_arg_indices [$current_residue get index] ;# Add indices of lysine atoms to list
            set dna_contact_atom [atomselect top "index $contact_dna"]
            set dna_residue_num [$dna_contact_atom get residue]
            if {$dna_residue_num in $processed_dna_residues} {
                continue
            }
            set dna_residue [atomselect top "residue $dna_residue_num"]
            set dna_residue_com [measure center $dna_residue weight mass]
            puts $output_file "Contact, $contact_residue-$contact_dna, $dna_residue_com"
            lappend processed_dna_residues $dna_residue_num

            # Append the points to the corresponding lists
            lappend lys_arg_points $center_of_mass
            lappend dna_points $dna_residue_com

        }
    }
}

# Reset the processed dna residues list
set processed_dna_residues {}

# Loop over all the arginine residues
foreach residue $arginine_residues {
    set current_residue [atomselect top "resname ARG and residue $residue"]
    set contacts [measure contacts 5 $current_residue $dna]
    if {[llength [lindex $contacts 0]] > 0 || [llength [lindex $contacts 1]] > 0} {
        puts "Arginine residue $residue is in contact with DNA"
        set center_of_mass [measure center $current_residue weight mass]
        puts $output_file "Arginine, $residue, $center_of_mass"
        foreach contact_residue [lindex $contacts 0] contact_dna [lindex $contacts 1] {
            lappend lys_arg_indices [$current_residue get index] ;# Add indices of lysine atoms to list
            set dna_contact_atom [atomselect top "index $contact_dna"]
            set dna_residue_num [$dna_contact_atom get residue]
            if {$dna_residue_num in $processed_dna_residues} {
                continue
            }
            set dna_residue [atomselect top "residue $dna_residue_num"]
            set dna_residue_com [measure center $dna_residue weight mass]
            puts $output_file "Contact, $contact_residue-$contact_dna, $dna_residue_com"
            lappend processed_dna_residues $dna_residue_num
        }
    }
}

# Close the file
close $output_file

# Define a graphics molecule to draw into
mol new
set graphics_mol [molinfo top]

# Now draw lines between each pair of points
for {set i 0} {$i < [llength $lys_arg_points]} {incr i} {
    set lys_arg_point [lindex $lys_arg_points $i]
    set dna_point [lindex $dna_points $i]
    
    # Convert each point to a VMD vector
    set lys_arg_vector [vecscale 1 $lys_arg_point]
    set dna_vector [vecscale 1 $dna_point]
    
    # Draw a line from the lysine/arginine residue to the DNA residue
    draw line $lys_arg_vector $dna_vector style solid width 1
}

# Hide everything in the original molecule
mol modselect 0 $mol_id "none"

# Show only the atoms involved in the contacts
mol addrep $mol_id
mol modselect 0 $mol_id "index [join $lys_arg_indices " or index "]"
mol modstyle 0 $mol_id Ribbons
mol addrep $mol_id
mol modselect 1 $mol_id "index [join $processed_dna_residues " or index "]"
mol modstyle 1 $mol_id Ribbons
