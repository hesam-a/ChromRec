import numpy as np
import matplotlib.pyplot as plt

# Given parameters
r_Pauli = 16.4  # Pauli radius in Angstroms
overlap_distance = r_Pauli - 0.2 * r_Pauli
r_values = np.linspace(0.1, 2 * r_Pauli, 1000)
delta_r = r_values - overlap_distance

# Potential function parameters (hypothetical for visualization)
a = 1  # amplitude
b = 4  # decay rate for exponential
c = 1  # offset for logarithmic and softened power potential
n = 6  # power for softened power potential
d = 1.5

# Exponential Potential
V_exp = a * np.exp(-b * delta_r)

# tanh Potential
V_tanh = -a * (np.tanh(d*(delta_r)) - 1)

# Logarithmic Potential
V_log = a * np.log((delta_r + c)/b ) * -V_tanh 
          
# Modified Gaussian Potential
V_gauss = a * (np.exp(-(delta_r**2)/2*b**2) - 1)

# inverse Polynomial Potential
V_invPol = (a/(delta_r)**2) - b

# Calculate potential values for each sigma
V_comb = V_exp * V_tanh/a

# Plot
plt.figure(figsize=(16, 10))

plt.plot(r_values, V_exp, label='Exponential Potential', color='blue', linewidth=2)
#plt.plot(r_values, V_log, label='Logarithmic Potential', color='green', linewidth=2)
#plt.plot(r_values, V_gauss, label='Modified Gaussian Potential', color='red', linewidth=2)
#plt.plot(r_values, V_tanh, label='Hyperbolic Tangent Potential', color='black', linewidth=2)
#plt.plot(r_values, V_invPol, label='Inverse Polynomial Potential', color='magenta', linewidth=2)
#plt.plot(r_values, V_comb, label=f'Tanh-Exp', color ='black',linewidth=2)
plt.axvline(x=r_Pauli, label=f"Pauli Radius ({r_Pauli} \u00C5)", color='black', linestyle='--', linewidth=2)
plt.axvline(x=overlap_distance, label=f"20% Overlap Point ({overlap_distance} Å)", color='red', linestyle='-.',  linewidth=2)
plt.title('Potential Functions Centered Around Pauli Radius', fontweight='bold', fontsize=12)
plt.xlabel('Distance (r) [Å]',fontweight='bold', fontsize=12, labelpad=12)
plt.ylabel('Potential Energy',fontweight='bold', fontsize=12, labelpad=12)
plt.legend(prop={'weight':'bold'})
plt.grid(True)
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)

plt.legend(prop={'weight':'bold'})
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylim([-5, 100])
plt.xlim([10, 17])
plt.tight_layout()
plt.show()
