# Load the PDB file
set pdb_file [lindex $argv 0]
mol new $pdb_file
#set dcd_file [lindex $argv 1]
#mol addfile $dcd_file

set nf [molinfo top get numframes]
set distance_list [list]
for {set i 0} {$i < $nf} {incr i} {
   [atomselect top "all"] frame $i
   set com_t1 [measure center [atomselect top "residue 30 or residue 31 or residue 32 or residue 33 or residue 34 or residue 35 or residue 36 or residue 37 or residue 38 or residue 39 or residue 40 or residue 41 or residue 42 or residue 43 or residue 44 or residue 329 or residue 330 or residue 331 or residue 332 or residue 333 or residue 334 or residue 335 or residue 336 or residue 337 or residue 338 or residue 339 or residue 340 or residue 341 or residue 342 or residue 343"]]
   set com_t2 [measure center [atomselect top "residue 45 or residue 46 or residue 47 or residue 48 or residue 49 or residue 50 or residue 51 or residue 52 or residue 53 or residue 54 or residue 55 or residue 56 or residue 57 or residue 58 or residue 59 or residue 314 or residue 315 or residue 316 or residue 317 or residue 318 or residue 319 or residue 320 or residue 321 or residue 322 or residue 323 or residue 324 or residue 325 or residue 326 or residue 327 or residue 328"]]
   set dist [veclength [vecsub $com_t1 $com_t2]]
   lappend distance_list $dist
}

set outfile [open "distance_list.txt" "w"]
foreach dist $distance_list {
    puts $outfile $dist
}
close $outfile

## Calculate mean distance
#set total 0
#foreach dist $distance_list {
#    set total [expr {$total + $dist}]
#}
#set mean [expr {$total / [llength $distance_list]}]
#
## Calculate variance of distances
#set total 0
#foreach dist $distance_list {
#    set dev [expr {$dist - $mean}]
#    set total [expr {$total + $dev*$dev}]
#}
#set variance [expr {$total / [llength $distance_list]}]

