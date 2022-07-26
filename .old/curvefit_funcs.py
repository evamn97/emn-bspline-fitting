# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 21:15:44 2020

Functions for fitting NURBS curves to input data using a fixed number of control points, and refining the knot
vectors to achieve better results.

@author: eva_n
"""

# imports
# from geomdl.visualization import VisMPL as vis
from math import sqrt
from geomdl import operations
import matplotlib.pyplot as plt
import numpy as np
from geomdl_mod import Mfitting, Mconstruct


# from mpl_toolkits.mplot3d import Axes3D

def unref_fit_fast(fastscan_datafile: str, profiles: int, deg=3, ctrlpts_size=30, plot=True):
    # TODO: add an input for "ideal"/"ground truth" datafile

    """ Fits fast-scan profile data to NURBS curves using a fixed number of control points and constructs a NURBS surface.

    :param fastscan_datafile: comma delimited csv filename of fast scan data in XYZ format.
    :type fastscan_datafile: str (ex: "stepspatt.csv")
    :param: profiles: the number of profile scans.
    :type profiles: int
    :param deg: degree of output curves. default=3
    :type deg: int
    :param ctrlpts_size: fixed number of control points to use in fitting routine. default=30
    :type ctrlpts_size: int

    """

    # load data_v
    data_v = np.loadtxt(fastscan_datafile, delimiter=',')
    data_v[:, [0, 1]] = data_v[:, [1, 0]] * 100  # assume z-unit = (x,y)-unitE-3 (scaling just for convenience)
    # Exporting to XYZ data_v from gwyddion gives you a left-handed coordinate system so I swap the x and y cols
    #   such that (x, y) => (u, v)

    # ??? is there a good way to extract the number of profiles from the data without having to enter it manually?

    # split the data into a dictionary of scan scans for curve fitting
    data_profiles = np.split(data_v, profiles)  # splits data_v into list of arrays. "profiles" sets the # of splits
    profile_dict = {}
    key = 0
    for scan in data_profiles:  # map to a tuple of tuples so it can be used in Mfitting
        profile_dict[key] = tuple(map(tuple, scan))  # maps each array in the list to a tuple of tuples
        key += 1

    # Fit NURBS curves_v to each scan line *** the profile scans are along the v-direction if (x, y) => (u, v) ***
    curves_v = []
    for key in profile_dict:  # using profile_dict because it's the tuples
        fit_pts = profile_dict[key]
        curve = Mfitting.approximate_curve(fit_pts, deg, ctrlpts_size=ctrlpts_size)
        curves_v.append(curve)

    # Check the residual error between curves and data
    # TODO: Create some kind of list or array of residuals to plot the error from "ideal" data
    max_err_list = []
    error_dict = {}
    index = 0
    while index < len(curves_v):
        evl_points = data_profiles[index]
        #    using data_profiles because it's an ndarray and I can call only y values easier
        evl_points_list = list(evl_points[:, 1])  # it's a curve along v so since y => v, use y values in evaluate_list
        evl_points_norm = []
        for point in evl_points_list:  # normalize the points values to between 0 and 1
            norm = (float(point) - min(evl_points_list)) / (max(evl_points_list) - min(evl_points_list))
            evl_points_norm.append(norm)
        curve = curves_v[index]
        crv_pts = np.array(curve.evaluate_list(evl_points_norm))

        residuals = []
        row = 0
        while row < len(evl_points):
            error = sqrt((evl_points[row, 0] - crv_pts[row, 0]) ** 2 + (evl_points[row, 1] - crv_pts[row, 1]) ** 2 + (
                    evl_points[row, 2] - crv_pts[row, 2]) ** 2)
            residuals.append(error)
            row += 1

        error_dict[index] = residuals
        max_err = np.amax(residuals)
        max_err_list.append(max_err)
        index += 1

    # Construct NURBS surface from curves_v *** the profile scans are along the v-direction if (x, y) => (u, v) ***
    surf_v = Mconstruct.construct_surface('v', curves_v, degree=deg)

    if plot:

        # Grab some middle curves_v' indices for plotting, and a sample of data_v points
        one_third = int(len(curves_v) / 3)
        two_thirds = int(len(curves_v) * (2 / 3))
        idx = np.random.randint(0, len(data_v), int(len(data_v) / 5))
        idx = np.unique(idx)

        # Plot the curves_v + surface alongside data_v points
        cv0pts = np.array(curves_v[0].evalpts)
        cv1_3pts = np.array(curves_v[one_third].evalpts)
        cv2_3pts = np.array(curves_v[two_thirds].evalpts)
        cv_1pts = np.array(curves_v[-1].evalpts)
        surfpts = np.array(surf_v.evalpts)
        datapts = data_v[idx, :]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(datapts[:, 0], datapts[:, 1], datapts[:, 2], color='red')
        ax.plot(cv0pts[:, 0], cv0pts[:, 1], cv0pts[:, 2], color='blue', linewidth=3.0)
        ax.plot(cv1_3pts[:, 0], cv1_3pts[:, 1], cv1_3pts[:, 2], color='blue', linewidth=3.0)
        ax.plot(cv2_3pts[:, 0], cv2_3pts[:, 1], cv2_3pts[:, 2], color='blue', linewidth=3.0)
        ax.plot(cv_1pts[:, 0], cv_1pts[:, 1], cv_1pts[:, 2], color='blue', linewidth=3.0)
        ax.plot_trisurf(surfpts[:, 0], surfpts[:, 1], surfpts[:, 2], color='orange', alpha=0.75, linewidth=4)
        plt.show()

    return data_v, curves_v, error_dict


def refine_fit_fast(fastscan_datafile: str, profiles: int, err_bound: float, width_add_knots: float,
                    num_add_knots: int, deg=3, ctrlpts_size=30):
    """ Inserts new knots into the knot vector of NURBS curves from unref_fit_fast() and re-fits to the fast-scan data
        using new knot vector iteratively until error threshold is satisfied or loops limit is reached.

    :param fastscan_datafile: comma delimited csv filename of fast scan data in XYZ format.
    :type fastscan_datafile: str (ex: "stepspatt.csv")
    :param: profiles: the number of profile scans.
    :type profiles: int
    :param err_bound: threshold for residual error, used for knot insertion.
    :type err_bound: float
    :param width_add_knots: distance in (u, v) scale within which to add knots on either side of high error.
    :type width_add_knots: float
    :param num_add_knots: number of new knots to add on either side of high error.
    :type num_add_knots: int
    :param deg: degree of output curves. default=3
    :type deg: int
    :param ctrlpts_size: fixed number of control points to use in fitting routine. default=30
    :type ctrlpts_size: int

    """

    # Get data as ndarray and unrefined curves fit
    data_v, curves_v, error_dict = unref_fit_fast(fastscan_datafile, profiles, deg, ctrlpts_size, plot=False)

    # split data_v into a dictionary of profile scans for curve fitting
    data_profiles = np.split(data_v, profiles)
    #   splits data_v array into list of arrays. "profiles" sets the # of splits
    profile_dict = {}
    key = 0
    for scan in data_profiles:  # map to a tuple of tuples so it can be used in Mfitting
        profile_dict[key] = tuple(map(tuple, scan))  # maps each array in the list to a tuple of tuples
        key += 1

    # Calculate the residuals between the curves and data points and if it's above err_bound, insert new unique
    #   knots into the knot vector and re-fit to the data.
    rcurves = []
    new_err_dict = {}

    # Loop through each curve and check error
    index = 0  # index is which curve we're looking at
    while index < len(curves_v):
        evl_points = data_profiles[index]
        #    using data_profiles because it's an ndarray and I can call only y values easier
        evl_points_list = list(evl_points[:, 1])  # it's a curve along v so since y => v, use y values in evaluate_list
        evl_points_norm = []
        for point in evl_points_list:  # normalize the points values to between 0 and 1
            norm = (float(point) - min(evl_points_list)) / (max(evl_points_list) - min(evl_points_list))
            evl_points_norm.append(norm)
        curve = curves_v[index]
        rcurve = None

        # Loop through the array of residuals for the curve and if error > err_bound, add knots. Make sure it's
        #   not an infinite loop by putting a condition on loops
        residuals = error_dict[index]
        max_err = np.amax(residuals)
        loops = 0
        while max_err > err_bound and loops < 10:  # "while" sort of used as an if statement here also
            new_knots = []

            # Check each residual in the array of residuals and see if it's greater than err_bound
            for i, error in enumerate(residuals):
                # i is the index of the list of residuals for each curve; it's the same as the index for evl_points.
                if error > err_bound:
                    add_knots = []
                    err_loc = evl_points_norm[i]  # u (or v) location of large error
                    add_knots_tmp = []
                    k_temp_a = np.linspace(err_loc - width_add_knots, err_loc, num=num_add_knots)
                    k_temp_b = k_temp_a + err_loc
                    add_knots_tmp.extend(k_temp_a.tolist())
                    add_knots_tmp.extend(k_temp_b.tolist())
                    add_knots_tmp = list(set(add_knots_tmp))
                    for k in add_knots_tmp:
                        if k not in curve.knotvector:
                            add_knots.append(k)
                    new_knots.extend(add_knots)

            new_knots = list(set(new_knots))
            new_knots.sort()
            rcurve_tmp = operations.insert_knot(curve, new_knots, [1])
            fit_pts = profile_dict[index]
            rcurve = Mfitting.approximate_curve(fit_pts, rcurve_tmp.degree, kv=rcurve_tmp.knotvector,
                                                ctrlpts_size=rcurve_tmp.ctrlpts_size)
            rcrv_pts = np.array(rcurve.evaluate_list(evl_points_norm))

            residuals = []
            row = 0
            while row < len(evl_points):
                error = sqrt(
                    (evl_points[row, 0] - rcrv_pts[row, 0]) ** 2 + (evl_points[row, 1] - rcrv_pts[row, 1]) ** 2 + (
                            evl_points[row, 2] - rcrv_pts[row, 2]) ** 2)
                residuals.append(error)
                row += 1

            max_err = np.amax(residuals)
            loops += 1

        new_err_dict[index] = residuals
        if rcurve is None:
            rcurves.append(curve)
        else:
            rcurves.append(rcurve)
        index += 1

    # Create the *refined* NURBS surface from the new cuvrves
    rsurf = Mconstruct.construct_surface('v', rcurves, degree=deg)

    # Grab some middle curves' indices and a sample of data for plotting
    one_third = int(len(rcurves) / 3)
    two_thirds = int(len(rcurves) * (2 / 3))
    idx = np.random.randint(0, len(data_v), int(len(data_v) / 5))
    idx = np.unique(idx)

    # Plot the refined curves alongside data points and new surface
    datapts = data_v[idx, :]
    rcv0pts = np.array(rcurves[0].evalpts)
    rcv1_3pts = np.array(rcurves[one_third].evalpts)
    rcv2_3pts = np.array(rcurves[two_thirds].evalpts)
    rcv_1pts = np.array(rcurves[-1].evalpts)
    rsurfpts = np.array(rsurf.evalpts)
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    ax2.scatter(datapts[:, 0], datapts[:, 1], datapts[:, 2], color='red')
    ax2.plot(rcv0pts[:, 0], rcv0pts[:, 1], rcv0pts[:, 2], color='blue', linewidth=3.0)
    ax2.plot(rcv1_3pts[:, 0], rcv1_3pts[:, 1], rcv1_3pts[:, 2], color='blue', linewidth=3.0)
    ax2.plot(rcv2_3pts[:, 0], rcv2_3pts[:, 1], rcv2_3pts[:, 2], color='blue', linewidth=3.0)
    ax2.plot(rcv_1pts[:, 0], rcv_1pts[:, 1], rcv_1pts[:, 2], color='blue', linewidth=3.0)
    ax2.plot_trisurf(rsurfpts[:, 0], rsurfpts[:, 1], rsurfpts[:, 2], color='orange', alpha=0.75, linewidth=4)

    plt.show()

    return rcurves, new_err_dict


# unref_fit_fast("stepspatt.csv", 35)
# refine_fit_fast("stepspatt.csv", 35, 2.2e-06, 0.04, 10)
