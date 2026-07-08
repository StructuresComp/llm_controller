from matplotlib import pyplot as plt
import numpy as np

#-----------------------------------------------------------------------------
# The following four functions define curvature profiles for the C, S, U, and L shapes.
# All the functions assume that the arclength (rod length) is 1.
# If that is not the case, divide the output (curvature) by the rod length to maintain the same shape.
#-----------------------------------------------------------------------------
# c-shape curvature profile: realistic C with terminal curls.
def kappa_c(s, kc0=4.5, Ac=3.0, sigc=0.15):
    # VERY IMPORTANT: IF ROD_LENGTH is not one, kappa_c should be divided by ROD_LENGTH
    return kc0 + Ac * (np.exp(-(s/sigc)**2) + np.exp(-((1.0-s)/sigc)**2))

# s-shape: two opposite bowls; this is the one from the corrected figure
def kappa_s(s, As=10.0, Aend=3.0, sig=0.30):
    base = As * np.sin(2.0 * np.pi * s)
    bottom_curl = np.exp(-(s / sig)**2)
    top_curl    = np.exp(-((1.0 - s) / sig)**2)
    return base + Aend * (bottom_curl - top_curl)

# U-shape
def kappa_u(s, width=0.65):
    scalar_input = np.isscalar(s)
    s = np.asarray(s, dtype=float)
    u = (s - 0.5) / (0.5 * width)
    bump = np.where(
        np.abs(u) <= 1.0,
        0.5 * (1.0 + np.cos(np.pi * u)),
        0.0,
    )
    A = 2.0 * np.pi / width
    out = A * bump
    if scalar_input:
        return float(out)
    return out

# L-shape
def kappa_l(s, s0=0.65, width=0.25):
    # s0 is the normalized arc-length location of the center of the L bend.
    # width is the normalized arc-length span over which the bend is smoothly distributed.
    scalar_input = np.isscalar(s)
    s = np.asarray(s, dtype=float)
    u = (s - s0) / (0.5 * width)
    bump = np.where(np.abs(u) <= 1.0,
                    0.5 * (1.0 + np.cos(np.pi * u)),
                    0.0)
    A = np.pi / width
    out = A * bump
    return float(out) if scalar_input else out
#-----------------------------------------------------------------------------

def main():
    #-----------------------------------------------------------------------------
    # The following two parameters should not affect the shape, only the size and the resolution
    ROD_LENGTH = 50.0
    nv = 80
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    kappa_function = kappa_s  # Change this to kappa_c, kappa_s, kappa_u, or kappa_l to visualize different shapes
    #-----------------------------------------------------------------------------

    ne = nv - 1
    deltaL = ROD_LENGTH / ne

    q = np.zeros( (nv, 2) )

    x0 = np.array([ 0, 0])
    x1 = np.array([deltaL, 0])
    e_i = x1 - x0
    q[0,:] = x0
    q[1,:] = x1

    for i in range(2, nv):
        s_current = (1.0 / ne ) * (i-1)
        e_i = q[i-1,:] - q[i-2,:]
        kappa_current = kappa_function(s_current) / ROD_LENGTH # Scale by rod length
        theta = 2.0 * np.arctan(kappa_current * deltaL / 2.0)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])

        e_f = R @ e_i
        q[i] = q[i-1] + e_f

    plt.figure(1)
    plt.plot(q[:,0], q[:,1], linewidth=4)
    plt.axis("equal")
    plt.show()

# Run it using: $ python src/tests/test_plot_shapes.py
if __name__ == "__main__":
    main()