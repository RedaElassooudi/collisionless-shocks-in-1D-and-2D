from math import sqrt


# Function that determines the optimal timestep based on the CFL condition and the plasma frequency condition
def calculate_dt_max(dx, v, qm_e, safety_factor=5):
    """
    Determine the maximal timestep based on the CFL and plasma frequency conditions:
        - dt < dx / v
        - dt < 2 / omega_p
    The minimal value is divided by safety_factor and returned.
    Wikipedia suggests a safety factor of 20 (for the second restriction only) for improved energy conservation
    """
    # TODO: i'm not fully convinced what the conditions on dt should be using normalized units

    # CFL condition (particle shouldn't cross more than one cell per timestep)
    dt_cfl = dx / v

    # Plasma frequency condition
    # TODO: likely wrong, w_p = sqrt((n * q^2) / (m * eps)) -> How to find n?
    wp = sqrt(abs(qm_e))  # Plasma frequency (normalized units)
    dt_wp = 2 / wp

    # Return the more restrictive timestep divided by the safety factor
    return min(dt_cfl, dt_wp) / safety_factor
