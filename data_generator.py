""" This program generates XYZ point cloud data for use in the 'apprx_fitting' program. The z height data_v is based on
    a 'line-space' pattern with a user-defined top surface height.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def create_data(width_mm: float, length_mm: float, n: int, total_points: int, sample_spacing_nm: float, patt_width_um: float, height_nm: float, edge_slope: float):
    """ Generates XYZ data_v samples from a line-space stepped pattern.
     

    :param width_mm: width (x-dir) in millimeters of area to be scanned.
    :param length_mm: length (y-dir) in millimeters of area to be scanned.
    :param n: number of profiles to scan.
    :param total_points: total number of desired data_v points.
    :param sample_spacing_nm: spacing between sample points in nms
    :param patt_width_um: width of one line or one space in line-space pattern (they are equal) in microns.
    :param height_nm: height (z-dir) of top surface of line in nanometers.
    :param edge_slope: slope of the rising/falling edge of the pattern
    :return: ndarray of XYZ data_v.
    
    """

    # I'm just converting everything to nanometers for convenience
    width_nm = width_mm * 1000000
    length_nm = length_mm * 1000000
    patt_width_nm = patt_width_um * 1000

    # Calculate the number of samples taken in one scan
    x_dim = int(total_points / n)
    # Calculate the number of line/space pairs & make sure it's an integer
    pairs = int(width_nm / (patt_width_nm * 2))

    # --------------------------------- Construct the pattern (1 point per nanometer) ---------------------------------

    # z pattern
    z_up = np.linspace(0, height_nm, int(height_nm), endpoint=False)
    z_top = np.tile([height_nm], int(patt_width_nm))
    z_down = np.flip(z_up)
    z_bottom = np.tile([0], int(patt_width_nm))
    z_patt_single = np.concatenate([z_up, z_top, z_down, z_bottom])
    # z_pattern = np.tile(z_patt_single, pairs)

    # x pattern
    x_up = np.tile([0], int(height_nm))
    x_top = np.linspace(0, patt_width_nm, int(patt_width_nm), endpoint=False)
    x_down = np.tile([patt_width_nm], int(height_nm))
    x_bottom = np.linspace(patt_width_nm, (2 * patt_width_nm), int(patt_width_nm), endpoint=False)
    x_patt_single = np.concatenate([x_up, x_top, x_down, x_bottom])

    # ----------------------------- Construct x-z pattern from a trapezoidal wave signal ------------------------------

    x = np.arange(0, width_nm, sample_spacing_nm)

    a = edge_slope * patt_width_nm * signal.sawtooth((2 * np.pi * x / patt_width_nm) - (patt_width_nm / 2), width=0.5) / 4.
    a[a > height_nm / 2.] = height_nm / 2.
    a[a < -height_nm / 2.] = -height_nm / 2.
    z = a + height_nm / 2.

    plt.scatter(x, z)
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    # Generate y values for each scan (y = interval*scan + y_start)
    interval = length_nm / n
    y_start = interval / 2
    y_singles = [y_start]
    num = 1
    while num < n:
        y = interval * num + y_start
        y_singles.append(y)
        num += 1
    y_values = np.repeat([y_singles], x_dim)
    #
    # Generate z values by sampling pattern.
    # The index "i" is used to call positions across the entire scanned area, but because the line-space pattern 
    #   repeats, I have only generated the first line and space in x and z for efficiency. Thus "i" will inevitably 
    #   surpass the last index of the generated pattern, so "i_s" is used to determine the equivalent index in 
    #   z_patt_single. For the x values, I add the full pattern width_add_knots (line and space together) multiplied by the 'pair'
    #   of line-space pairs the index position "i" would belong to if the pattern was generated across the whole area. 
    # "i" is rounded when called as an index because delta is a float variable, and for my purposes it is not important
    #   if the position is off by less than a nanometer. 
    delta = (len(z_patt_single) * pairs) / x_dim
    i = delta / 2
    z_profile = None
    while i < len(z_patt_single) * pairs:
        p = int(round(i) / len(z_patt_single))
        i_s = round(i) - (p * len(z_patt_single))
        z = z_patt_single[i_s]
        if z_profile is None:
            z_profile = z
        else:
            z_profile = np.append(z_profile, z)
        i += delta
    z_values = np.tile(z_profile, n)

    # Generate x values by sampling pattern
    delta = (len(x_patt_single) * pairs) / x_dim
    i = delta / 2
    x_profile = None
    while i < len(x_patt_single) * pairs:
        p = int(round(i) / len(x_patt_single))
        i_s = round(i) - (p * len(x_patt_single))
        x = (x_patt_single[i_s] + (p * 2 * patt_width_nm))
        if x_profile is None:
            x_profile = x
        else:
            x_profile = np.append(x_profile, x)
        i += delta
    x_values = np.tile(x_profile, n)

    xyz_data = np.column_stack((x_values, y_values, z_values))
    # pattern = np.column_stack((x_patt_single, z_patt_single))

    return xyz_data
    # return pattern
    # return len(x_patt_single), len(z_patt_single)


width = .002    # mm
length = .1   # mm
profiles = 10
points = 1000
spacing = 12.  # nm
pattern = .45  # um
height = 100.   # nm
slope = 1.5
results = create_data(width, length, profiles, points, spacing, pattern, height, slope)

# plot z_patt_single
ax = plt.axes(projection='3d')
ax.scatter(results[:, 0], results[:, 1], results[:, 2], color="red")
# ax.scatter(results[:, 0], results[:, 1], color="red")

plt.show()
