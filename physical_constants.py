q_e = -1  # Normalized electron charge
q_i = 1  # Normalized ion charge
m_e = 1  # Normalized electron mass
m_i = 1836  # Normalized ion mass
# TODO: set to physical values with normalized units
v_te = 1.0  # Thermal velocity for electrons
v_ti = 0.1  # Thermal velocity for ions
eps_0 = 1  # vacuum permitivity
#if we set speed of ligth equal to 1 we find that vacuum permeability will also be 1 for eps_0 = 1
c = 1 #should be changed but mu and eps should be carefully considered
mu_0 = 1 / (eps_0 * c^2) #vacuum permeability in dimensinless units
