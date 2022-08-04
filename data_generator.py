""" This program generates XYZ point cloud data for use in the 'apprx_fitting' program. The z height data_v is based on
    a 'line-space' pattern with a user-defined top surface height.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal, ndimage


def xz_plot(X_line, Z_line, title="AFM Scan Pattern"):
    title += ", # points = {}".format(len(X_line))
    plt.rc('font', size=20)  # controls default text sizes
    fig = plt.figure(figsize=(21, 10))
    ax = plt.axes()
    ax.plot(X_line, Z_line, label="X-Z scan plane")
    ax.set_xlabel("X-dir (width, nm)")
    ax.set_ylabel("Z-dir (height, nm)")
    ax.set_title(title)
    fig.tight_layout()
    ax.legend(loc="upper right")
    plt.show()


def create_data(size: tuple, n: int, sample_spacing_nm: float, patt_width_um: float, edge_slope: float, noise_sigma: float):
    """ Generates XYZ data_v samples from a line-space stepped pattern.
     
    :param size: tuple (w, l, h) specifying width (x-dir, millimeters), length (y-dir, millimeters), height (z-dir, nanometers) of scan area
    :param n: number of profiles to scan
    :param sample_spacing_nm: spacing between sample points in nms
    :param patt_width_um: width of one line or one space in line-space pattern (they are equal) in microns.
    :param edge_slope: slope of the rising/falling edge of the pattern
    :param noise_sigma: standard deviation of the noise added to the pattern
    :return: ndarray of XYZ data.
    """

    # I'm just converting everything to nanometers for convenience
    width_nm = size[0] * 1000000
    length_nm = size[1] * 1000000
    height_nm = size[2]
    patt_width_nm = patt_width_um * 1000

    # ----------------------------- Construct x-z pattern from a trapezoidal wave signal -------------------------------

    x_single = np.arange(0, width_nm, sample_spacing_nm)

    a = edge_slope * patt_width_nm * signal.sawtooth((2 * np.pi * x_single / patt_width_nm), width=0.5) / 4.
    a[a > height_nm / 2.] = height_nm / 2.
    a[a < -height_nm / 2.] = -height_nm / 2.
    z_single = a + height_nm / 2.

    x_values = np.tile(x_single, n)
    z_values = np.tile(z_single, n)
    if noise_sigma > 0:
        s = np.random.default_rng().normal(scale=noise_sigma, size=len(z_values))
        z_values = np.add(z_values, s)

    # ------------------------------------- Generate y values for each scan --------------------------------------------

    y_start = (length_nm / n) / 2

    y_singles = np.linspace(y_start, length_nm, n)
    y_values = np.repeat([y_singles], len(x_single))

    # -------------------------------------------- Put it together -----------------------------------------------------

    xyz_data = np.stack((x_values, y_values, z_values), axis=1)

    return xyz_data


width = 0.2    # mm     (1 um = .001 mm)
length = .01   # mm
height = 25.   # nm
profiles = 8
sample_spacing = 20.  # nm
pattern_pitch = 10.  # um
slope = 1.75
noise_stdev = 0.75
results = create_data((width, length, height), profiles, sample_spacing, pattern_pitch, slope, noise_stdev)

np.savetxt("data/lines_patt3.csv", results, delimiter=",")

X, Y, Z = results[:, 0].reshape((profiles, -1)), results[:, 1].reshape((profiles, -1)), results[:, 2].reshape((profiles, -1))

# %% Plotting
xz_plot(X[0], Z[0])

# filter_size = 38
# # small_filter = int(filter_size / 5)
# small_filter = int(np.sqrt(filter_size))
# Z_filtered = ndimage.median_filter(Z[0], size=small_filter, mode='nearest')
# # xz_plot(X[0], Z_filtered, title="Median filter")
# Z_filtered2 = ndimage.median_filter(Z_filtered, size=filter_size, mode='nearest')
# xz_plot(X[0], Z_filtered2, title="Median filter, size={}".format(filter_size))

