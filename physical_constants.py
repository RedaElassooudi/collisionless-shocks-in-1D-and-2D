from scipy.constants import k, e, epsilon_0, mu_0, m_e, m_p, c, G

# q_e = -1  # Normalized electron charge
# q_i = 1  # Normalized ion charge
# m_e = 1  # Normalized electron mass
# m_i = 1836  # Normalized ion mass
# # TODO: set to physical values with normalized units
# v_te = 0.000001  # Thermal velocity for electrons
# v_ti = 0.0000005  # Thermal velocity for ions
# eps_0 = 1  # vacuum permitivity
# #if we set speed of ligth equal to 1 we find that vacuum permeability will also be 1 for eps_0 = 1
# c = 1 #should be changed but mu and eps should be carefully considered
# mu_0 = 1 / (eps_0 * c^2) #vacuum permeability in dimensinless units

q_e = -e  # Normalized electron charge
q_i = e  # Normalized ion charge
m_e = m_e  # Normalized electron mass
m_i = m_p  # Normalized ion mass
# TODO: set to physical values with normalized units
v_te = 1000000  # Thermal velocity for electrons
v_ti = 20000  # Thermal velocity for ions
eps_0 = epsilon_0  # vacuum permitivity
#if we set speed of ligth equal to 1 we find that vacuum permeability will also be 1 for eps_0 = 1
c = c #should be changed but mu and eps should be carefully considered
# mu_0 = 1 / (eps_0 * c^2) #vacuum permeability in dimensinless units
mu_0 = mu_0
