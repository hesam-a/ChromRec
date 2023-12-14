import numpy as np
import scipy.constants


########## k_B T unit ##########

#           kg m^2 K      kg m^2 
#  k_B T = ---------- =  --------
#             s^2 K         s^2 

########## 𝜂 unit ##########

#                  kg s        kg
# 𝜂 = Pa * S = ----------- = -------  
#                 m s^2        m s

########## D_A unit ##########

#             k_B * T                  kg m^2            m^2
# D_A  = ------------------ = --------------------- = ---------
#         6 * 𝜋 * 𝜂 * r_A         s^2    kg     m         s
#                                      -------
#                                        m s

# calculate D_A based on a given radius and a viscosity (𝜂, here it's cell lysate's)
def calculate_D_A(eta, radius):

    #D_A = (scipy.constants.Boltzmann * scipy.constants.convert_temperature(25, 'celsius', 'kelvin'))/(6 * np.pi * eta * radius) 
    D_A = (1.380649e-23 * 298)/(6 * np.pi * eta * radius) 
    return D_A

# calculate RMSD based on the D_A and time step
def calculate_RMSD(D_A, time_step):
    return np.sqrt(2 * D_A * time_step)

#eta       = 0.00312      # cell lysate's viscosity
eta       = 0.02     
#radius    = 14.4e-10
radius    = 18.489e-10
time_step = 10e-9

D_A = calculate_D_A(eta, radius)

RMSD = calculate_RMSD(D_A, time_step)

print(f"\n     D_A  = {D_A:.4e}   m^2/s\n     RMSD = {RMSD:.4e}   meters\n")
