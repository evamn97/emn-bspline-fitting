"""
    Fits input data set to NURBS curves within error threshold and constructs a NURBS surface from the refined curves.
"""

# imports
from typing import Union
from geomdl import construct, operations, helpers
# from geomdl.visualization import VisMPL as vis
from geomdl_mod import Mfitting, Mconstruct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import sqrt
import pandas as pd
import pathos.multiprocessing as mp
from pathos.pools import ProcessPool as Pool


def gen_curve(single_curve_pts, deg, cp_size):  # generate a curve fit with fixed num of ctpts
    from geomdl_mod import Mfitting     # pathos raises an 'MFitting undefined' error without this
    fit_pts_arr = single_curve_pts.values
    fit_pts = list(map(tuple, fit_pts_arr))
    curve = Mfitting.approximate_curve(fit_pts, deg, ctrlpts_size=cp_size)
    curve.delta = 0.001
    return curve


def parallel_fitting(arr_split, deg, cp_size):
    in_deg = np.repeat(deg, len(arr_split))
    in_cp = np.repeat(cp_size, len(arr_split))
    pool = Pool(mp.cpu_count())
    curves_out = pool.map(gen_curve, arr_split, in_deg, in_cp)
    return curves_out


def get_error(profile_pts: pd.DataFrame, curve_obj):
    """ Gets the fitting error for a single curve.
    :param profile_pts: dataframe of profile points
    :param curve_obj: fitted curve object
    :return: fitted curve error, ndarray
    """
    import numpy as np  # pathos raises an 'np undefined' error without this
    if not type(profile_pts) == np.ndarray:
        points = profile_pts.values  # get array of points from df
    else:
        points = profile_pts
    eval_idx = list(np.linspace(0, 1, len(points)))
    curve_points = np.array(curve_obj.evaluate_list(eval_idx))
    curve_error = np.sqrt(np.sum(np.square(points - curve_points), axis=1))
    return curve_error


def parallel_errors(arr_split, curves):
    pool = Pool(mp.cpu_count())
    error = pool.map(get_error, arr_split, curves)
    return error


def curve_plotting(data_xyz, crv):
    """ Plots a single curve in the X-Z plane
    :param data_xyz: x-z data points, pandas df
    :param crv: curve to plot, NURBS.Curve
    :return: none
    """

    crv_pts = np.array(crv.evalpts)
    ct_pts = np.array(crv.ctrlpts)
    crv_err = get_error(data_xyz, crv)

    fig, ax = plt.subplots(2, figsize=(20, 15))
    ax[0].grid(True)
    ax[0].plot(data_xyz.values[:, 0], data_xyz.values[:, 2], label='Input Data', c='blue', linewidth=1.5, marker='.', markersize=4)
    ax[0].plot(crv_pts[:, 0], crv_pts[:, 2], label='Fitted Curve', c='red', alpha=0.7)
    ax[0].plot(ct_pts[:, 0], ct_pts[:, 2], label='Control Points', marker='+', c='orange', linestyle='--', linewidth=0.75)
    ax[0].legend()
    ax[0].set(xlabel='Lateral Position X [nm]', ylabel='Height [nm]', title='B-spline Result')

    ax[1].grid(True)
    ax[1].plot(data_xyz.values[:, 0], crv_err, 'k', label='Fitting error')
    ax[1].legend()
    ax[1].set(xlabel='Lateral Position X [nm]', ylabel='Error [nm]', title='Fitting Error')

    plt.show()


def main():
    # input set by the user
    deg_u = 3
    deg_v = 3
    ctrlpts_size_u = 30
    ctrlpts_size_v = 30
    err_bound = 4.8e-06
    width_add_knots = 0.04  # for knot refinement/insertion
    num_add_knots = 10  # number of knots to add on EACH SIDE of the high error

    # load data
    filename = "../data/lines_patt.csv"
    data_2d = pd.read_csv(filename, delimiter=',', names=['x', 'y', 'z'])

    # dimensions and inputs inferred from data
    profiles = len(set(data_2d['y'].values))  # y-values on a single profile are the same, so profiles is number of unique y-values
    tot_points = len(data_2d)
    y_dim = int(tot_points / profiles)
    size_u = profiles
    size_v = y_dim

    # data_3d = np.reshape(data_2d.values, (profiles, -1, 3))   # for if I want a 3D array later
    data_profiles = np.array_split(data_2d, profiles)
    curves_v = parallel_fitting(data_profiles, deg_v, ctrlpts_size_v)

    # --------------------------------------------- Plot a single curve for reference -------------------------------

    i = np.random.randint(0, len(data_profiles))
    c = curves_v[i]
    # curve_err = get_error(data_profiles[i], c)
    data_pts = data_profiles[i]
    curve_plotting(data_pts, c)

    # ---------------------------------------------- Refitting ------------------------------------------------------
    # Calculate the residuals between the curves and data points and if it's above err_bound, insert new unique knots
    #   into the knot vector and re-fit to the data.
    rcurves = []
    max_err_list = []
    max_density = 3
    new_err_list = []
    index = 0
    while index < len(curves_v):
        evl_points = data_profiles[index]  # using data_profiles because it's an ndarray and I can call only y values easier
        evl_points_list = list(evl_points[:, 1])  # it's a curve along v so since y => v, use y values in evaluate_list
        evl_points_norm = []
        for point in evl_points_list:  # normalize the points values to between 0 and 1
            norm = (float(point) - min(evl_points_list)) / (max(evl_points_list) - min(evl_points_list))
            evl_points_norm.append(norm)
        curve = curves_v[index]
        crv_pts = np.array(curve.evaluate_list(evl_points_norm))
        # residuals = evl_points[:, 2] - crv_pts[:, 2]
        residuals = []
        row = 0
        while row < len(evl_points):
            error = sqrt(
                (evl_points[row, 0] - crv_pts[row, 0]) ** 2 + (evl_points[row, 1] - crv_pts[row, 1]) ** 2 + (
                        evl_points[row, 2] - crv_pts[row, 2]) ** 2)
            residuals.append(error)
            row += 1
        max_err = np.amax(residuals)
        max_err_list.append(max_err)
        rcurve = None
        if max_err > err_bound:
            density = 1
            while max_err > err_bound and density < max_density:
                ctpts = curve.ctrlptsw if curve.rational else curve.ctrlpts
                rctpts, rkv = helpers.knot_refinement(curve.degree, curve.knotvector, ctpts, density=density)
                fit_pts = profile_dict[index]
                rcurve = Mfitting.approximate_curve(fit_pts, curve.degree, kv=rkv, ctrlpts_size=len(rctpts))
                rcrv_pts = np.array(rcurve.evaluate_list(evl_points_list))
                # residuals = evl_points[:, 2] - rcrv_pts[:, 2]
                residuals = []
                row = 0
                while row < len(evl_points):
                    error = sqrt(
                        (evl_points[row, 0] - rcrv_pts[row, 0]) ** 2 + (evl_points[row, 1] - rcrv_pts[row, 1]) ** 2 + (
                                evl_points[row, 2] - rcrv_pts[row, 2]) ** 2)
                    residuals.append(error)
                    row += 1
                max_err = np.amax(residuals)
                density += 1
        if rcurve is None:
            rcurves.append(curve)
        else:
            rcurves.append(rcurve)
        new_err_list.append(max_err)
        index += 1

    # Create the *refined* NURBS surface
    # rsurf = Mconstruct.construct_surface('v', rcurves, degree=3)

    # Plot the *refined* curves_v + surface alongside data_v points
    rcv0pts = np.array(rcurves[0].evalpts)
    rcv1_3pts = np.array(rcurves[one_third].evalpts)
    rcv2_3pts = np.array(rcurves[two_thirds].evalpts)
    rcv_1pts = np.array(rcurves[-1].evalpts)
    # rsurfpts = np.array(rsurf.evalpts)
    fig2 = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.scatter(datapts[:, 0], datapts[:, 1], datapts[:, 2], color='red')
    ax2.plot(rcv0pts[:, 0], rcv0pts[:, 1], rcv0pts[:, 2], color='blue', linewidth=3.0)
    ax2.plot(rcv1_3pts[:, 0], rcv1_3pts[:, 1], rcv1_3pts[:, 2], color='blue', linewidth=3.0)
    ax2.plot(rcv2_3pts[:, 0], rcv2_3pts[:, 1], rcv2_3pts[:, 2], color='blue', linewidth=3.0)
    ax2.plot(rcv_1pts[:, 0], rcv_1pts[:, 1], rcv_1pts[:, 2], color='blue', linewidth=3.0)
    # ax2.plot_trisurf(rsurfpts[:, 0], rsurfpts[:, 1], rsurfpts[:, 2], color='orange', alpha=0.75, linewidth=4)

    # plt.show()
    plt.savefig("refined_fit.png")


if __name__ == '__main__':
    main()

# TODO: fit the two scan directions to surf_u and surf_v, then take the fitted control points in u for surf_u,
#  and in v for surf_v, and apply them to an overall surf_v
#  TODO: take the fitted curves_v in both directions and measure difference between them at crossing points
