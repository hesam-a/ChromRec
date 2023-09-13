import numpy as np
import matplotlib.pyplot as plt

# Given parameters for the van der Waals Gaussian potential
r_vdW = 16.4  # vdW radius in Angstroms (for visualization, representing one bead)
r_values = np.linspace(0.1, (2 * r_vdW)+50, 1000)

# Parameters for the vdW Gaussian potential
a_vdW = 1  # amplitude (negative for attraction)
b_vdW = 15  # this will make sure that by around 10 nm the potential is almost zero

# Gaussian potential function
V_gauss_vdW = -a_vdW * np.exp(-(r_values - r_vdW)**2 / (2 * b_vdW**2))

# Plotting the potential
plt.figure(figsize=(10, 6))
plt.plot(r_values, V_gauss_vdW, label="vdW Gaussian Potential", color='blue', linewidth=2)

# Adding the equation to the plot
equation  = r"$V_{vdW}(r) = -A \times exp(- \frac{(r - r_{vdW})^2}{2\sigma^2})$"
equation += f"\n$A={a_vdW}$,"+" $r_{vdW}=$"+f"${r_vdW}$, $\sigma={b_vdW}$"
r_descrip = "r = distance between the center of mass of each bead"
plt.text(54, -0.57, equation, bbox=dict(facecolor='white', alpha=0.5), fontsize=16, color='purple')#, fontweight='bold')
plt.text(54, -0.63, r_descrip, bbox=dict(facecolor='white', alpha=0.5), fontsize=10, color='purple')#, fontweight='bold')

plt.axvline(x=r_vdW, color='red', linestyle='--', label=f"r_vdW = {r_vdW} Å", linewidth=2)
plt.axvline(x=r_vdW-0.2*r_vdW, color='magenta', linestyle='-.', label=f"overlap = {r_vdW-0.2*r_vdW} Å", linewidth=2)
plt.title("vdW Gaussian Potential for One Bead", fontweight='bold', fontsize=12)
plt.xlabel("Distance between the centers of mass (Å)", fontweight='bold', fontsize=12)
plt.ylabel("Potential V(r)", fontweight='bold', fontsize=12)
plt.legend(prop={'weight':'bold'})
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylim([-1.05, 0.05])
plt.xlim([0, 100])
plt.grid(True)
plt.tight_layout()
plt.show()
