import sys
import numpy as np
from scipy.stats import norm
import scipy.constants as sc
import matplotlib.pyplot as plt

# Convert your distance list to a numpy array
#distances = np.loadtxt('distance_list.txt')
#distances = np.loadtxt('angles_degree.txt')
distances = np.loadtxt(sys.argv[1])

# Fit a normal distribution to the data
mu, std = norm.fit(distances)
cos_angles = np.cos(distances)
ExpVal_cos_theta = np.mean(cos_angles)
print(f"expectation value of cos(theta): {ExpVal_cos_theta}")
p_l = 100/np.log(-ExpVal_cos_theta)
print(f" persistence length from 3 beads (10 nm long): {p_l}")


# Plot the histogram
plt.hist(distances, bins=25, density=True, alpha=0.6, color='g')


# Calculate temperature in Kelvin
T = 298  # replace with your temperature in Kelvin

# Boltzmann constant in kcal/mol/K
k_B = sc.Boltzmann * sc.Avogadro / 1000  # 0.0019872041 kcal/mol/K

# Calculate the spring constant
k = k_B * T / (std**2)

print(f'Spring constant:  {k:.3f}')

# Plot the PDF
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel("Angles in deg")
title = "Angle fit results: mu = %.2f deg,  std = %.2f" % (mu, std)
plt.title(title)

plt.show()
