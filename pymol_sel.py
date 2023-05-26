from pymol import cmd

# Load the PDB file
pdb_file = 'pan_dna.pdb'
cmd.load(pdb_file)

# Define selections
sel1 = 'resi 30+31+32+33+34+35+36+37+38+39+40+41+42+43+44+329+330+331+332+333+334+335+336+337+338+339+340+341+342+343'
sel2 = 'resi 45+46+47+48+49+50+51+52+53+54+55+56+57+58+59+314+315+316+317+318+319+320+321+322+323+324+325+326+327+328'
sel3 = 'resi 60+61+62+63+64+65+66+67+68+69+70+71+72+73+74+299+300+301+302+303+304+305+306+307+308+309+310+311+312+313'

# Create named selections
cmd.select('sel1', sel1)
cmd.select('sel2', sel2)
cmd.select('sel3', sel3)

# Show these selections
cmd.show('spheres', 'sel1 or sel2 or sel3')

