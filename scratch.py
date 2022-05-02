import numpy as np
import pandas as pd
import pathos.multiprocessing as mp
from pathos.pools import ProcessPool as Pool
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from geomdl_mod import Mfitting


def gen_curve(single_curve_pts, deg, cp_size):  # generate a curve fit with fixed num of ctpts
    from geomdl_mod import Mfitting  # pathos raises an 'MFitting undefined' error without this
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


def get_error(profile_pts, curve_obj):
    """ Gets the fitting error for a single curve.
    :param profile_pts: dataframe of profile points
    :param curve_obj: fitted curve object
    :return: fitted curve error, ndarray
    """
    import numpy as np  # pathos raises an 'np undefined' error without this
    points = profile_pts.values  # get array of points from df
    eval_idx = list(np.linspace(0, 1, len(points)))
    curve_points = np.array(curve_obj.evaluate_list(eval_idx))
    curve_error = np.sqrt(np.sum(np.square(points - curve_points), axis=1))
    return curve_error


def parallel_errors(arr_split, curves):
    pool = Pool(mp.cpu_count())
    error = pool.map(get_error, arr_split, curves)
    return error


def find_nearest(a, a0):    # not used
    """Element index in nd array `a` closest to the scalar value `a0`"""
    idx = np.abs(a - a0).argmin()
    return idx


def normalize(arr): # not used
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def curve_plotting(data_xz, crv):
    """ Plots a single curve in the X-Z plane
    :param data_xz: x-z data points, pandas df
    :param crv: curve to plot, NURBS.Curve
    :return: none
    """

    crv_pts = np.array(crv.evalpts)
    ct_pts = np.array(crv.ctrlpts)
    crv_err = get_error(data_xz, crv)

    fig, ax = plt.subplots(2, figsize=(20, 15))
    ax[0].grid(True)
    ax[0].plot(data_xz.values[:, 0], data_xz.values[:, 1], label='Input Data', c='blue', linewidth=1.5, marker='.', markersize=4)
    ax[0].plot(crv_pts[:, 0], crv_pts[:, 2], label='Fitted Curve', c='red', alpha=0.7)
    ax[0].plot(ct_pts[:, 0], ct_pts[:, 2], label='Control Points', marker='+', c='orange', linestyle='--', linewidth=0.75)
    ax[0].legend()
    ax[0].set(xlabel='Lateral Position X [nm]', ylabel='Height [nm]', title='B-spline Result')

    ax[1].grid(True)
    ax[1].plot(data_xz.values[:, 0], crv_err, 'k', label='Fitting error')
    ax[1].legend()
    ax[1].set(xlabel='Lateral Position X [nm]', ylabel='Error [nm]', title='Fitting Error')

    plt.show()


if __name__ == '__main__':
    filename = "lines_patt.csv"
    data_2d = pd.read_csv(filename, delimiter=',', names=['x', 'y', 'z'])
    profiles = len(set(data_2d['y'].values))
    deg = 3
    cpts_size = 50

    arr_splitting = np.array_split(data_2d, profiles)
    curves_fit = parallel_fitting(arr_splitting, deg, cpts_size)

    # curve_plotting(arr_splitting, curves_fit)

    i = np.random.randint(0, len(arr_splitting))
    c = curves_fit[i]
    curve_err = get_error(arr_splitting[i], c)
    data_pts = arr_splitting[i][['x', 'z']].values
    curve_plotting(data_pts, c)

    # cv0pts = np.array(curves_fit[0].evalpts)
    # cv1_3pts = np.array(curves_fit[3].evalpts)
    # cv2_3pts = np.array(curves_fit[6].evalpts)
    # cv_1pts = np.array(curves_fit[-1].evalpts)
    #
    # fig = plt.figure(figsize=(12.8, 9.6))
    # ax = plt.axes(projection='3d')
    # ax.plot(data_2d[:, 0], data_2d[:, 1], data_2d[:, 2], color='red')
    # ax.plot(cv0pts[:, 0], cv0pts[:, 1], cv0pts[:, 2], color='blue', linewidth=3.0)
    # ax.plot(cv1_3pts[:, 0], cv1_3pts[:, 1], cv1_3pts[:, 2], color='blue', linewidth=3.0)
    # ax.plot(cv2_3pts[:, 0], cv2_3pts[:, 1], cv2_3pts[:, 2], color='blue', linewidth=3.0)
    # ax.plot(cv_1pts[:, 0], cv_1pts[:, 1], cv_1pts[:, 2], color='blue', linewidth=3.0)
    # plt.show()
