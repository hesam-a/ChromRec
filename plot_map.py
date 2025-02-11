import sys
import numpy as np
import matplotlib.pyplot as plt

#data = np.loadtxt("contact_map_oe.txt", skiprows=1) 
data = np.loadtxt(sys.argv[1], skiprows=1) 
# skiprows=1 because the first line has n_bins n_bins

plt.figure(figsize=(10,8))

im = plt.imshow(np.log1p(data), cmap='Reds', interpolation='nearest')
#im = plt.imshow(np.log1p(data), cmap='Reds', interpolation='gaussian')
#im = plt.imshow(data, cmap='Reds', interpolation='nearest')
cbar = plt.colorbar(im)
cbar.set_label(label='log(1 + O/E)', size=16)
plt.title("O/E Contact Map", fontsize=32)
plt.xlabel('Bin index',  fontsize=15)
plt.ylabel('Bin index',  fontsize=15)
plt.title('Hi-C O/E Contact Map')
plt.show()
