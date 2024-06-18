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
    ax.plot(X_line/1000, Z_line, label="X-Z scan plane")
    ax.set_xlabel("X-dir (width, um)")
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


if __name__ == '__main__':
    # 1 mm       =  1e+0 mm  =  1000 um  =  1e+3 um  =  1000000 nm   =  1e+6 nm
    # .001 mm    =  1e-3 mm  =     1 um  =  1e+0 um  =     1000 nm   =  1e+3 nm
    # .000001 mm =  1e-6 mm  =  .001 um  =  1e-3 um  =        1 nm   =  1e+0 nm

    width_x = 0.02      # millimeters
    length_y = 0.02     # millimeters

    patt_pitch = 5.     # microns
    height_z = 20.      # nanometers
    slope = 1.
    noise_stdev = 0.5

    profiles = 1
    # total_points = 1000

    # for total_points in [500, 1000, 5000, 10000, 20000, 50000, 100000]:
    for total_points in [2500]:
        sample_spacing = (width_x * 1000000) / total_points

        for n in [0, noise_stdev]:
            results = create_data((width_x, length_y, height_z), profiles, sample_spacing, patt_pitch, slope, n)

            noisy = '_noisy' if n > 0 else '_clean'
            # np.savetxt(f"data_new/lines_{total_points}{noisy}.csv", results, delimiter=",")

            X, Y, Z = results[:, 0].reshape((profiles, -1)), results[:, 1].reshape((profiles, -1)), results[:, 2].reshape((profiles, -1))
            xz_plot(X[0], Z[0])

    # %% Plotting
    # filter_size = 38
    # small_filter = int(filter_size / 5)
    # small_filter = int(np.sqrt(filter_size))
    # Z_filtered = ndimage.median_filter(Z[0], size=small_filter, mode='nearest')
    # xz_plot(X[0], Z_filtered, title="Median filter")
    # Z_filtered2 = ndimage.median_filter(Z_filtered, size=filter_size, mode='nearest')
    # xz_plot(X[0], Z_filtered2, title="Median filter, size={}".format(filter_size))

