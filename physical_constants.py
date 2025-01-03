q_e = -1  # Normalized electron charge
q_i = 1  # Normalized ion charge
m_e = 1  # Normalized electron mass
m_i = 1836  # Normalized ion mass
# TODO: set to physical values with normalized units
v_te = 1e-6  # Thermal velocity for electrons
v_ti = 5e-7  # Thermal velocity for ions
eps_0 = 1  # vacuum permitivity
# if we set speed of ligth equal to 1 we find that vacuum permeability will also be 1 for eps_0 = 1
c = 1  # should be changed but mu and eps should be carefully considered
mu_0 = 1 / (eps_0 * c**2)  # vacuum permeability in dimensinless units

# Using these units we have the following other important units
# Length L: 1 = 3.54112824901163e-14 m  = mu_0 * e^2 / m_e
# Time t: 1 = 1.181193240362181e-22 s = mu_0 * e^2 / (m_e * c)
# Energy E: 1 = 8.187105775475753e-14 J (kg m^2 s^-2) = c * c * m_e
