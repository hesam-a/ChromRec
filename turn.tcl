# Load the PDB file
set pdb_file [lindex $argv 0]
mol new $pdb_file

# Calculate the number of turns
set num_turns [expr {int(374/30) + 1}]

for {set turn_num 0} {$turn_num < $num_turns} {incr turn_num} {
    set forw_start_res [expr {($turn_num) * 15}]
    set forw_end_res [expr {$forw_start_res + 14}]
    set rev_start_res [expr {373 - ($turn_num) * 15}]
    set rev_end_res [expr {$rev_start_res - 14}]
    puts "$forw_start_res   $forw_end_res   $rev_end_res   $rev_start_res"

    # Construct selection string
    set sel_str ""
    for {set i $forw_start_res} {$i <= $forw_end_res} {incr i} {
        append sel_str "residue $i "
        if {$i != $forw_end_res} {
            append sel_str "or "
        }
    }
    append sel_str "or "
    for {set i $rev_end_res} {$i <= $rev_start_res} {incr i} {
        append sel_str "residue $i "
        if {$i != $rev_start_res} {
            append sel_str "or "
        }
    }

    puts $sel_str
    set sel [atomselect top $sel_str]
    puts $sel
#
#
#    # Check if the selection has atoms
#    if {[$sel num] > 0} {
#        set padded_turn_num [format "%02d" [expr {$turn_num + 1}]]
#        $sel writepdb "turn_${padded_turn_num}.pdb"
#    } else {
#        puts "No atoms found for turn $turn_num"
#    }
    
}

