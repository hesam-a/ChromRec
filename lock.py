import numpy as np
import matplotlib.pyplot as plt

hist_radius = 18.496
# Gaussian parameters
a = -1      # amplitude
b = hist_radius - 0.6 * hist_radius  # center (at radius)
c = 2      # width 
d = 4      # width 

# the distance between the sites. Generate x values from 0 to 2*histone_radius Å
x = np.linspace(0, 2*hist_radius, 20*int(hist_radius))

# Compute the Gaussian
V_gauss1 = a * np.exp(-(x - b)**2 / (2 * c**2))
V_gauss2 = a * np.exp(-(x - b)**2 / (2 * d**2))
V_sum    = a * ( np.exp(-(x - b)**2 / (2 * c**2)) + np.exp(-(x - b)**2 / (2 * d**2)))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(x, V_gauss1, label='Sharp Gaussian',   color='blue',    linewidth=2)
plt.plot(x, V_gauss2, label='Broaden Gaussian', color='#ff0096', linewidth=2)
plt.plot(x, V_sum,    label='Sum of Gaussians', color='magenta', linewidth=2)
plt.axvline(x=b,    linestyle='--', color='red',     label=f'Centered at Histone Radius {round(b,2)} Å',linewidth=2)
plt.axvline(x=b+5,  linestyle='--', color='blue',    label=f'5 Å from Histone Radius',linewidth=2)
plt.axvline(x=b+10, linestyle='--', color='#ff0096', label=f'10 Å from Histone Radius',linewidth=2)
plt.xlabel('Distance (Å) between sites', fontweight='bold', fontsize=12, labelpad=12)
plt.ylabel('Potential Value', fontweight='bold', fontsize=12, labelpad=12)
plt.title('Sharp Gaussian for Lock-and-Key Potential', fontweight='bold', fontsize=12)
plt.legend(prop={'weight':'bold'})
plt.grid(True)
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
#plt.ylim([-100, 5])
#plt.xlim([10, 17])
plt.tight_layout()
plt.show()
