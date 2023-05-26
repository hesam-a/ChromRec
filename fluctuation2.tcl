# Load the PDB file
set pdb_file [lindex $argv 0]
mol new $pdb_file type pdb
set dcd_file [lindex $argv 1]
mol addfile $dcd_file type dcd waitfor all

set nf [molinfo top get numframes]
puts "Number of frames: $nf"   ;# Let's print out the number of frames

set distance_list [list]
   set sel1 [atomselect top "residue 9 or residue 364"]
   set sel2 [atomselect top "residue 24 or residue 349"]

for {set i 0} {$i < $nf} {incr i} {
   $sel1 frame $i
   $sel2 frame $i
   set com_t1 [measure center $sel1]
   set com_t2 [measure center $sel2]
   set dist [veclength [vecsub $com_t1 $com_t2]]
   lappend distance_list $dist
   puts "Frame $i distance: $dist"   ;# Let's print out the distance for each frame
}

set outfile [open "distance_list_linker.txt" "w"]
foreach dist $distance_list {
    puts $outfile $dist
}
close $outfile

