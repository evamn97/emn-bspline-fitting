""" This program generates XYZ point cloud data for use in the 'apprx_fitting' program. The z height data_v is based on
    a 'line-space' pattern with a user-defined top surface height.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal


def create_data(width_mm: float, length_mm: float, height_nm: float, n: int, sample_spacing_nm: float, patt_width_um: float, edge_slope: float):
    """ Generates XYZ data_v samples from a line-space stepped pattern.
     

    :param width_mm: width (x-dir) in millimeters of area to be scanned.
    :param length_mm: length (y-dir) in millimeters of area to be scanned.
    :param height_nm: height (z-dir) of top surface of line in nanometers.
    :param n: number of profiles to scan.
    :param sample_spacing_nm: spacing between sample points in nms
    :param patt_width_um: width of one line or one space in line-space pattern (they are equal) in microns.
    :param edge_slope: slope of the rising/falling edge of the pattern
    :return: ndarray of XYZ data_v.
    
    """

    # I'm just converting everything to nanometers for convenience
    width_nm = width_mm * 1000000
    length_nm = length_mm * 1000000
    patt_width_nm = patt_width_um * 1000

    # ----------------------------- Construct x-z pattern from a trapezoidal wave signal -------------------------------

    x_single = np.arange(0, width_nm, sample_spacing_nm)

    a = edge_slope * patt_width_nm * signal.sawtooth((2 * np.pi * x_single / patt_width_nm), width=0.5) / 4.
    a[a > height_nm / 2.] = height_nm / 2.
    a[a < -height_nm / 2.] = -height_nm / 2.
    z_single = a + height_nm / 2.

    x_values = np.tile(x_single, n)
    z_values = np.tile(z_single, n)

    # ------------------------------------- Generate y values for each scan --------------------------------------------

    # (y = interval*scan + y_start)
    y_start = (length_nm / n) / 2

    y_singles = np.linspace(y_start, length_nm, n)
    y_values = np.repeat([y_singles], len(x_single))

    # -------------------------------------------- Put it together -----------------------------------------------------

    xyz_data = np.stack((x_values, y_values, z_values), axis=1)

    return xyz_data


width = .002    # mm
length = .1   # mm
height = 50.   # nm
profiles = 10
spacing = 12.  # nm
pattern = .45  # um
slope = 2.
results = create_data(width, length, height, profiles, spacing, pattern, slope)

np.savetxt("lines_patt.csv", results, delimiter=",")

X, Y, Z = results[:, 0].reshape((profiles, -1)), results[:, 1].reshape((profiles, -1)), results[:, 2].reshape((profiles, -1))

fig = plt.figure(figsize=(12.8, 9.6))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.6, cmap=cm.plasma)
ax.scatter(X, Y, Z, alpha=0.25)
ax.set_xlabel("X-dir (width, nm)")
ax.set_ylabel("Y-dir (length, nm")
ax.set_title("AFM Scan Height (nm)")

plt.show()
