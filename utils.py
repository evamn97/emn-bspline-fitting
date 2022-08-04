import copy
from typing import Union

import numpy as np
import pandas as pd
import pathos.multiprocessing as mp
from geomdl import helpers, BSpline, fitting
from geomdl.exceptions import GeomdlException
from matplotlib import pyplot as plt
from pathos.pools import ProcessPool as Pool
from scipy import ndimage as nd

from geomdl_mod import Mfitting, Moperations as Mop


def find_nearest(a, a0):
    """
    Finds element array `a` closest to the scalar value `a0`.
    Returns index position (and value) of result.
    """
    a = np.asarray(a)
    idx = np.abs(a - a0).argmin()
    value = a[idx]
    return idx, value


def normalize(arr: Union[list, np.ndarray], low=0., high=1.) -> np.ndarray:
    arr = np.asarray(arr)
    unit_norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    result = (unit_norm * (high - low)) + low
    return result


def get_curve_section(curve, profile_pts: pd.DataFrame, bounds: np.ndarray):
    """
    Returns a section of curve defined by low and high u_k values, with associated fitting points.
    :param curve: curve object to split
    :param profile_pts: data points the input curve is fitted to
    :param bounds: boundary u_k values
    :return: specified curve section (with clamped exterior knots), section data points
    """
    from geomdl_mod import Mfitting, Moperations as Mop
    import numpy as np
    import pandas as pd  # pathos raises an 'MFitting undefined' error without this

    u_k = Mfitting.compute_params_curve(list(map(tuple, profile_pts.values)))
    low = bounds[0]
    high = bounds[1]

    if not all((low >= curve.knotvector[0], high <= curve.knotvector[-1])):
        raise ValueError("One of the bounds is outside the range of input curve's knot vector.")
    if low == 0 or low == curve.knotvector[0]:
        id_u, u_i = find_nearest(u_k, high)
        section_curve = Mop.split_curve(curve, u_i)[0]
        section_pts = np.array_split(profile_pts, [id_u + 1])[0]
        section_uk = np.array_split(u_k, [id_u + 1])[0]
    elif high == 1 or high == curve.knotvector[-1]:
        id_u, u_i = find_nearest(u_k, low)
        section_curve = Mop.split_curve(curve, u_i)[-1]
        section_pts = np.array_split(profile_pts, [id_u])[-1]
        section_uk = np.array_split(u_k, [id_u])[-1]
    else:
        id_u0, u_i0 = find_nearest(u_k, low)
        id_u1, u_i1 = find_nearest(u_k, high)
        section_curve = Mop.split_curve(Mop.split_curve(curve, u_i0)[-1], u_i1)[0]
        section_pts = np.array_split(profile_pts, [id_u0, id_u1 + 1])[1]
        section_uk = np.array_split(u_k, [id_u0, id_u1 + 1])[1]

    section_pts = pd.DataFrame(section_pts, columns=['x', 'y', 'z'])
    return section_curve, section_pts, section_uk


# def get_section_multi(curve, profile_pts: pd.DataFrame, splits: int):
#     """
#         Returns a list of curve sections defined by low and high u_k values, with associated fitting points.
#         :param curve: curve object to split
#         :param profile_pts: data points the input curve is fitted to
#         :param splits: how many sections to return
#         :return: list of curve section (with clamped exterior knots), list of section data points
#         """
#     u_k = Mfitting.compute_params_curve(list(map(tuple, profile_pts.values)))
#     temp = np.array_split(u_k, splits)
#     u_i = [u[0] for u in temp][1:]
#
#     for u_i in splits_ui:



def merge_curves(c1, c2):
    """
    Merges two curves into a single curve

    :param c1: first curve
    :param c2: second curve
    :return: merged curve
    """
    if not c1.knotvector[-1] == c2.knotvector[0]:
        raise ValueError("The input curves must intersect at the merge point.")
    if not c1.degree == c2.degree:
        raise ValueError("The input curves must be of the same degree.")

    p = c1.degree
    merged_curve = BSpline.Curve(normalize_kv=False)
    ctrlpts_new = c1.ctrlpts + c2.ctrlpts

    join_knot = c1.knotvector[-1]  # last knot of c1 == first not of c2, multiplicity s = 2 * p + 1 => needs to be s = 1
    kv_new = list(np.array(c1.knotvector + c2.knotvector).astype(np.float64))
    s = helpers.find_multiplicity(join_knot, kv_new)
    num_knots = len(ctrlpts_new) + p + 1  # assuming all cpts are kept, find the required number of knots from m = n + p + 1
    while len(kv_new) - num_knots > 0:
        kv_new.remove(join_knot)

    merged_curve.degree = p
    merged_curve.ctrlpts = ctrlpts_new
    merged_curve.knotvector = kv_new

    s = helpers.find_multiplicity(join_knot, kv_new)
    if s > p:
        merged_curve = Mop.remove_knot(merged_curve, [join_knot], [s - p])

    return merged_curve


def merge_curves_multi(args):
    """
    Merges two or more curves into a single curve

    :param args: curves to merge
    :return: merged curve
    """
    if len(args) < 2:
        raise ValueError("At least two curves must be specified in args")

    p = args[0].degree
    merged_curve = BSpline.Curve(normalize_kv=False)
    ctrlpts_new = []
    join_knots = []
    kv_new = []
    s = []

    for c in args:
        ctrlpts_new = ctrlpts_new + c.ctrlpts
        kv_new = sorted(kv_new + c.knotvector)
        join_knots.append(c.knotvector[-1])
    join_knots = join_knots[:-1]
    for j in join_knots:
        s_i = helpers.find_multiplicity(j, kv_new)
        for r in range(s_i - p - 1):    # ensures rule m = n + p + 1
            kv_new.remove(j)
        s_i = helpers.find_multiplicity(j, kv_new)
        s.append(s_i)

    kv_new = list(np.asarray(kv_new).astype(float))

    num_knots = len(ctrlpts_new) + p + 1  # assuming all cpts are kept, find the required number of knots from m = n + p + 1
    if num_knots != len(kv_new):
        raise ValueError("Something went wrong with getting the merged knot vector. Check knot removals.")

    merged_curve.degree = p
    merged_curve.ctrlpts = ctrlpts_new
    merged_curve.knotvector = kv_new
    merged_curve.delta = 0.001
    merged_curve.evaluate()

    where_s = np.where(np.asarray(s) > p)[0]    # returns a 1-dim tuple for some reason
    for i in where_s:
        # delete = (s[i] - 1)
        delete = s[i] - 2       # ex: s >= 4 => 4 - 3 + 1 = 2 deletes
        if delete > 0:
            merged_curve = Mop.remove_knot(merged_curve, [join_knots[i]], [delete])

    return merged_curve


# def adding_knots(profile_pts, curve, num, error_bound_value, u_k):
#     """
#     Adds num knots to the curve IF it reduces the fitting error.
#     :param profile_pts: profile data points
#     :type profile_pts: pandas.DataFrame
#     :param curve: curve to plot
#     :type curve: BSpline.Curve
#     :param num: number of knots to add
#     :type num: int
#     :param error_bound_value: maximum error value (nm) allowed
#     :type error_bound_value: float
#     :param u_k: parametric coordinates of profile_pts
#     :type u_k: numpy.ndarray
#     :return: refined curve
#     """
#     from geomdl_mod import Moperations as Mop, Mfitting
#     import numpy as np  # pathos raises an 'undefined' error without this
#
#     e_i = get_error(profile_pts, curve)
#     changed = False
#     if np.amax(e_i) < error_bound_value:
#         print("Section meets error bound")
#         return curve, changed
#
#     knots_i = curve.knotvector
#     # if we don't want to add more knots, refit using existing knot vector
#     if len(knots_i) + num >= int(len(profile_pts) / 2) or num == 0:
#         new_kv = curve.knotvector
#         temp_kv = list(normalize(new_kv))
#         try:
#             rfit_curve = Mfitting.approximate_curve(list(map(tuple, profile_pts.values)), curve.degree, kv=temp_kv)
#         except ValueError:
#             print("Cannot refit section")
#             return curve, changed
#         rfit_curve_err = get_error(profile_pts, rfit_curve)
#
#         if np.average(rfit_curve_err) < np.average(e_i):
#             rcrv = BSpline.Curve(normalize_kv=False)
#             rcrv.degree = curve.degree
#             rcrv.ctrlpts = rfit_curve.ctrlpts
#             rcrv.knotvector = new_kv
#             curve = rcrv
#             changed = True
#             # print("Section refit with num=0")
#         return curve, changed
#     kns = []
#     ei_max = np.amax(e_i)
#     kns.append(u_k[np.where(e_i == ei_max)[0][0]])
#     if num > 1:
#         ids = np.where((e_i > error_bound_value) & (e_i != ei_max))[0]
#
#     # kns = np.linspace(knots_i[0], knots_i[-1], num + 2)[1:-1]
#     duplicates = np.where(np.isin(kns, knots_i))[0]
#     for i in duplicates:
#         s = helpers.find_multiplicity(kns[i], knots_i)
#         if s >= curve.degree:
#             kns = np.delete(kns, i)
#
#     if len(kns) == 0:
#         # print("No new unique knots")
#         return curve, changed
#
#     new_kv = sorted(curve.knotvector + kns)
#     temp_kv = list(normalize(new_kv))
#     try:
#         rfit_curve = Mfitting.approximate_curve(list(map(tuple, profile_pts.values)), curve.degree, kv=temp_kv)
#     except ValueError:
#         print("Cannot refit section with new kv")
#         return curve, changed
#     rfit_curve_err = get_error(profile_pts, rfit_curve)
#
#     # print("e_i: {}, \nrfit: {}\n".format(np.average(e_i), np.average(rfit_curve_err)))
#
#     if np.average(rfit_curve_err) < np.average(e_i):
#         rcrv = BSpline.Curve(normalize_kv=False)
#         rcrv.degree = curve.degree
#         rcrv.ctrlpts = rfit_curve.ctrlpts
#         rcrv.knotvector = new_kv
#         curve = rcrv
#         changed = True
#         # print("Refit Section")
#
#     return curve, changed


def adding_knots(profile_pts, curve, num, error_bound_value, u_k):
    """
    Adds num knots to the curve IF it reduces the fitting error.
    :param profile_pts: profile data points
    :type profile_pts: pandas.DataFrame
    :param curve: curve to plot
    :type curve: BSpline.Curve
    :param num: number of knots to add
    :type num: int
    :param error_bound_value: maximum error value (nm) allowed
    :type error_bound_value: float
    :param u_k: parametric coordinates of profile_pts
    :type u_k: numpy.ndarray
    :return: refined curve
    """
    from geomdl_mod import Mfitting
    import numpy as np  # pathos raises an 'undefined' error without this

    e_i = get_error(profile_pts, curve, uk=u_k)
    ei_max = np.amax(e_i)
    changed = False

    if ei_max > error_bound_value or num == 0:
        knots_i = curve.knotvector
        kns = []
        # if we don't want to add more knots, refit using existing knot vector
        if len(knots_i) + num >= int(len(profile_pts)):
            # kns.append(np.linspace(knots_i[0], knots_i[-1], num + 2)[1:-1][0])
            ids2 = list(np.argsort(e_i)[::-1][num:num + 1])[0]
            kns = list(set(kns + list(u_k[ids2])))
            num = 0

        if num >= 1:
            ids = list(np.argsort(e_i)[::-1][:num])
            kns = list(set(kns + list(u_k[ids])))
            for k in kns:
                s = helpers.find_multiplicity(k, knots_i)
                if s < curve.degree:
                    continue
                if k == knots_i[0]:      # can't insert end point knots, so we shift over by one
                    repl_knot_idx = find_nearest(u_k, k)[0] + 1
                    i_kns = np.where(kns == k)[0][0]
                    kns[i_kns] = u_k[repl_knot_idx]     # use next largest u_k value
                elif k == knots_i[-1]:
                    repl_knot_idx = find_nearest(u_k, k)[0] - 1
                    i_kns = np.where(kns == k)[0][0]
                    kns[i_kns] = u_k[repl_knot_idx]  # use next largest u_k value
                elif s >= curve.degree:
                    kns.remove(k)

        new_kv = list(np.asarray(knots_i + kns).astype(float))
        new_kv.sort()
        temp_kv = list(normalize(new_kv))
        try:
            rfit_curve = Mfitting.approximate_curve(list(map(tuple, profile_pts.values)), curve.degree, kv=temp_kv)
        except ValueError:
            print("Cannot refit section with new kv")
            return curve, changed
        rfit_err = get_error(profile_pts, rfit_curve)

        if np.average(rfit_err) < np.average(e_i):
            rcrv = BSpline.Curve(normalize_kv=False)
            rcrv.degree = curve.degree
            rcrv.ctrlpts = rfit_curve.ctrlpts
            rcrv.knotvector = new_kv
            curve = rcrv
            ei_max = np.amax(get_error(profile_pts, rcrv, uk=u_k))
            changed = True
            # print("Refit Section")

    return curve, changed


def adding_knots2(profile_pts, curve, num, error_bound_value, u_k):
    """
    Adds num knots to the curve IF it reduces the fitting error.
    :param profile_pts: profile data points
    :type profile_pts: pandas.DataFrame
    :param curve: curve to plot
    :type curve: BSpline.Curve
    :param num: number of knots to add
    :type num: int
    :param error_bound_value: maximum error value (nm) allowed
    :type error_bound_value: float
    :param u_k: parametric coordinates of profile_pts
    :type u_k: numpy.ndarray
    :return: refined curve
    """
    from geomdl_mod import Moperations as Mop, Mfitting
    import numpy as np  # pathos raises an 'undefined' error without this

    e_i = get_error(profile_pts, curve, uk=u_k)
    ei_max = np.amax(e_i)
    changed = False

    while ei_max > error_bound_value:
        knots_i = curve.knotvector
        kns = []
        # if we don't want to add more knots, refit using existing knot vector
        if len(knots_i) + num >= int(len(profile_pts) / 2):
            num = 0

        if num >= 1:
            kns.append(u_k[np.where(e_i == ei_max)[0][0]])
            ids = list(np.where((e_i > error_bound_value) & (e_i != ei_max))[0])
            kns = kns + list(u_k[ids[:num-1]])
            duplicates = np.where(np.isin(kns, knots_i))[0]
            for i in duplicates:
                s = helpers.find_multiplicity(kns[i], knots_i)
                if s >= curve.degree:
                    kns = list(np.delete(kns, i))

        new_kv = sorted(curve.knotvector + kns)
        temp_kv = list(normalize(new_kv))
        try:
            rfit_curve = Mfitting.approximate_curve(list(map(tuple, profile_pts.values)), curve.degree, kv=temp_kv)
        except ValueError:
            print("Cannot refit section with new kv")
            break
        rfit_emax = np.amax(get_error(profile_pts, rfit_curve, uk=u_k))

        if rfit_emax < ei_max:
            rcrv = BSpline.Curve(normalize_kv=False)
            rcrv.degree = curve.degree
            rcrv.ctrlpts = rfit_curve.ctrlpts
            rcrv.knotvector = new_kv
            e_i = get_error(profile_pts, rcrv, uk=u_k)
            ei_max = np.amax(get_error(profile_pts, rcrv, uk=u_k))
            curve = rcrv
            changed = True
            # print("Refit Section")

        if num == 0:
            # only run once if there are no knots added
            break

    return curve, changed


def get_error(profile_pts, curve_obj, uk=None, sep=False):
    """ Gets the fitting error for a single curve.

    :param profile_pts: dataframe of profile points
    :param curve_obj: fitted curve object
    :param uk: parametric coordinates
    :param sep: if True, returns a 2D array of x, y, z error values
    :return: fitted curve error, ndarray
    """
    import numpy as np  # pathos raises an 'np undefined' error without this
    points = profile_pts.values  # get array of points from df
    if uk is not None:
        eval_idx = uk
    else:
        eval_idx = Mfitting.compute_params_curve(list(map(tuple, points)))
        if list(np.asarray(curve_obj.knotvector)[[0, -1]]) != [0.0, 1.0]:
            eval_idx = normalize(eval_idx, low=curve_obj.knotvector[0], high=curve_obj.knotvector[-1])
        eval_idx = list(eval_idx)
    curve_points = np.array(curve_obj.evaluate_list(eval_idx))
    curve_error = np.sqrt(np.sum(np.square(np.subtract(points, curve_points)), axis=1))
    if sep:
        x_error = np.sqrt(np.square(np.subtract(points[:, 0], curve_points[:, 0])))
        y_error = np.sqrt(np.square(np.subtract(points[:, 1], curve_points[:, 1])))
        z_error = np.sqrt(np.square(np.subtract(points[:, 2], curve_points[:, 2])))
        curve_error = np.stack((curve_error, x_error, y_error, z_error), axis=1)
    return curve_error


def parallel_errors(arr_split, curves):
    pool = Pool(mp.cpu_count())
    error = pool.map(get_error, arr_split, curves)
    return error


def curve_plotting(profile_pts, crv, error_bound_value, sep=False, uk=None, med_filter=0, filter_plot=False, title="BSpline curve fit plot"):
    """ Plots a single curve in the X-Z plane with corresponding fitting error.

    :param profile_pts: profile data points
    :type profile_pts: pandas.DataFrame
    :param crv: curve to plot
    :type crv: BSpline.Curve
    :param error_bound_value: defined error bound for iterative fit as ratio of maximum curve value
    :type error_bound_value: float (must be between 0 and 1)
    :param sep: if True, plot the x, y and z error in error plot
    :type sep: bool
    :param uk: parametric coordinates of the data points
    :type uk: list
    :param med_filter: sets median filter window size if not None
    :type med_filter: float
    :param filter_plot: whether to plot the filtered curve
    :param title: title of the figure
    :type title: string
    :return: none
    """
    # font sizes
    SMALL_SIZE = 26
    MEDIUM_SIZE = 30
    LARGE_SIZE = 34

    plt.rc('font', size=MEDIUM_SIZE)       # controls default text sizes
    plt.rc('axes', titlesize=LARGE_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=LARGE_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

    crv.delta = 0.001
    crv.evaluate()
    crv_pts = np.array(crv.evalpts)
    ct_pts = np.array(crv.ctrlpts)
    data_xz = profile_pts[['x', 'z']].values

    fig, ax = plt.subplots(2, figsize=(30, 20), sharex='all')
    if med_filter > 1:
        filtered_data = profile_pts[['x', 'y']]
        small_filter = int(np.sqrt(med_filter) + 1) if int(np.sqrt(med_filter)) >= 1 else 1
        filtered_z = nd.median_filter(nd.median_filter(nd.median_filter(profile_pts['z'].values,
                                                                        size=small_filter, mode='nearest'),
                                                       size=med_filter, mode='nearest'),
                                      size=small_filter, mode='nearest')
        filtered_data['z'] = filtered_z
        crv_err = get_error(filtered_data, crv, uk=uk, sep=sep)

        # ax[0].plot(filtered_data['x'].values, filtered_data['z'].values, label='Median Filtered Data', c='purple', linewidth=2)

        if filter_plot:
            fig2, ax2 = plt.subplots(figsize=(24, 14))
            ax2.plot(data_xz[:, 0], data_xz[:, 1], label='Input Data', c='blue', linewidth=0.7)
            ax2.plot(filtered_data['x'].values, filtered_data['z'].values, label='Median Filtered Data', c='purple', linewidth=2)
            ax2.grid(True)
            ax2.legend(loc="upper right")
            ax2.set(xlabel='Lateral Position X [nm]', ylabel='Height Z [nm]')
            fig2.suptitle('Median Filter Result'.upper())
            fig2.tight_layout()
            # plt.savefig("noisyfit_figures/med-fit.png")
    else:
        crv_err = get_error(profile_pts, crv, uk=uk, sep=sep)

    ax[0].grid(True)
    ax[0].plot(data_xz[:, 0], data_xz[:, 1], label='Input Data', c='blue', linewidth=1.5, marker='.', markersize=2)
    ax[0].plot(crv_pts[:, 0], crv_pts[:, 2], label='Fitted Curve', c='red', linewidth=2)
    ax[0].plot(ct_pts[:, 0], ct_pts[:, 2], label='Control Points', marker='+', c='orange', linestyle='--', linewidth=0.75)

    scaled_kv = normalize(crv.knotvector, low=np.amin(profile_pts['x']), high=np.amax(profile_pts['x']))
    ax[0].hist(scaled_kv, bins=(int(len(profile_pts['x']))), bottom=(np.amin(profile_pts['z']) - 10), label='Knot Locations')

    ax[0].set_ylim(np.amin(profile_pts['z'].values) - 10, np.amax(profile_pts['z'].values) + 10)
    ax[0].set(ylabel='Height Z [nm]',
              title=(title.upper() + ': Control Points={}, Data Size={}'.format(crv.ctrlpts_size, len(profile_pts))))

    ax[1].grid(True)
    if sep:
        ax[1].plot(data_xz[:, 0], crv_err[:, 0], 'k', label='Fitting error', linewidth=1.75)
        ax[1].plot(data_xz[:, 0], crv_err[:, 1], label='X error', c='blue', alpha=0.5)
        # ax[1].plot(data_xz[:, 0], crv_err[:, 2], label='Y error', c='green', alpha=0.3)
        ax[1].plot(data_xz[:, 0], crv_err[:, 3], label='Z error', c='red', alpha=0.7)
    else:
        ax[1].plot(data_xz[:, 0], crv_err, 'k', label='Fitting error')

    ax[1].axhline(y=error_bound_value, xmin=data_xz[0, 0], xmax=data_xz[-1, 0], color='k', linestyle='--',
                  label='Error bound', linewidth=1.75)
    ax[1].set(xlabel='Lateral Position X [nm]', ylabel='Error [nm]',
              title='Fitting Error: Max={}, Avg={}, Fitting Bound={} nm'.format(round(np.amax(crv_err), 4),
                                                                                round(np.average(crv_err), 4),
                                                                                round(error_bound_value, 2)))

    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    fig.tight_layout()

    # ax[0].legend(loc=(1.01, 0.5))
    # ax[1].legend(loc=(1.01, 0.5))
    # box = ax[0].get_position()
    # ax[0].set_position([box.x0 - 0.065, box.y0, box.width, box.height])
    # box = ax[1].get_position()
    # ax[1].set_position([box.x0 - 0.065, box.y0, box.width, box.height])

    fig_title = "noisyfit_figures/" + title.replace(' ', '-').lower() + "-cp{}".format(crv.ctrlpts_size) + ".png"
    # plt.savefig(fig_title)

    plt.show()


def plot_curve3d(profile_pts, curve, title="3D Curve Plot"):
    """
    Plots the curve of interest in a 3D profile.

    :param profile_pts: The 3D profile points
    :type profile_pts: pd.DataFrame
    :param curve: The 3D curve
    :param title: title string
    :return: None
    """
    X = profile_pts['x'].values
    Y = profile_pts['y'].values
    Z = profile_pts['z'].values
    curve.delta = 0.001
    curve.evaluate()
    curve_points = np.asarray(curve.evalpts)
    Cx = curve_points[:, 0]
    Cy = curve_points[:, 1]
    Cz = curve_points[:, 2]

    fig = plt.figure()
    fig.tight_layout()
    ax = plt.axes(projection='3d')
    fig.add_axes(ax)
    ax.scatter(X, Y, Z, label="Data Points", c='blue')
    ax.plot(Cx, Cy, Cz, label="BSpline Curve", c='red')
    ax.set(xlabel='Lateral Position X [nm]', ylabel='Lateral Position Y [nm]', zlabel='Height [nm]', title=title)

    plt.show()

