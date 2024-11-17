# Modify x, v (but possibly also E, B) based on boundary conditions


def apply_damping(x, v, x_boundary, width, factor=0.1):
    # differentiate between the layers at x = 0 and x = x_max
    if width > 0:
        damping_region = x <= x_boundary + width
        damp_factor = factor * min(v) / (x_boundary - width) ** 2
        v[damping_region] += abs(damp_factor * (x[damping_region] - width) ** 2)
    else:
        damping_region = x >= x_boundary + width
        damp_factor = factor * max(v) / (x_boundary - width) ** 2
        v[damping_region] -= abs(damp_factor * (x[damping_region] - width) ** 2)
    return
