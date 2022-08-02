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


def section_curve(curve, profile_pts: pd.DataFrame, bounds: Union[tuple, list]):
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
        new_curve = Mop.split_curve(curve, u_i)[0]
        section_pts = np.array_split(profile_pts, [id_u])[0]
    elif high == 1 or high == curve.knotvector[-1]:
        id_u, u_i = find_nearest(u_k, low)
        new_curve = Mop.split_curve(curve, u_i)[-1]
        section_pts = np.array_split(profile_pts, [id_u])[-1]
    else:
        id_u0, u_i0 = find_nearest(u_k, low)
        id_u1, u_i1 = find_nearest(u_k, high)
        new_curve = Mop.split_curve(Mop.split_curve(curve, u_i0)[-1], u_i1)[0]
        section_pts = np.array_split(profile_pts, [id_u0, id_u1 + 1])[1]

    section_pts = pd.DataFrame(section_pts, columns=['x', 'y', 'z'])
    return new_curve, section_pts


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
        kv_new = kv_new + c.knotvector
        if c.knotvector[-1] != 1.0:
            join_knots.append(c.knotvector[-1])
    for j in join_knots:
        s_i = helpers.find_multiplicity(j, kv_new)
        for r in range(s_i - p - 1):    # ensures rule m = n + p + 1
            kv_new.remove(j)
        s_i = helpers.find_multiplicity(j, kv_new)
        s.append(s_i)

    kv_new = list(np.asarray(kv_new).astype(float))

    where_s = np.where(np.asarray(s) > 1)[0]    # returns a 1-dim tuple for some reason

    num_knots = len(ctrlpts_new) + p + 1  # assuming all cpts are kept, find the required number of knots from m = n + p + 1
    if num_knots != len(kv_new):
        raise ValueError("Something went wrong with getting the merged knot vector. Check knot removals.")

    merged_curve.degree = p
    merged_curve.ctrlpts = ctrlpts_new
    merged_curve.knotvector = kv_new

    for i in where_s:
        merged_curve = Mop.remove_knot(merged_curve, [join_knots[i]], [s[i] - 1])

    return merged_curve


def adding_knots(profile_pts, curve, num, error_bound_value):
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
    :return: refined curve
    """
    from geomdl_mod import Moperations as Mop, Mfitting
    import numpy as np  # pathos raises an 'undefined' error without this

    e_i = get_error(profile_pts, curve)
    changed = False
    if np.amax(e_i) < error_bound_value:
        print("Section meets error bound")
        return curve, changed

    knots_i = curve.knotvector
    if len(knots_i) + num >= int(len(profile_pts) / 2) or num == 0:
        k = ((knots_i[-1] - knots_i[0]) / 2) + knots_i[0]   # get a centered knot
        new_kv = curve.knotvector
        new_kv.append(k)
        new_kv.sort()
        temp_kv = list(normalize(new_kv))
        try:
            rfit_curve = Mfitting.approximate_curve(list(map(tuple, profile_pts.values)), curve.degree, kv=temp_kv)
        except ValueError:
            print("Cannot refit section")
            return curve, changed
        rfit_curve_err = get_error(profile_pts, rfit_curve)

        if np.average(rfit_curve_err) < np.average(e_i):
            rcrv = BSpline.Curve(normalize_kv=False)
            rcrv.degree = curve.degree
            rcrv.ctrlpts = rfit_curve.ctrlpts
            rcrv.knotvector = new_kv
            curve = rcrv
            changed = True
            print("Section refit with num=1")
        return curve, changed

    kns = np.linspace(knots_i[0], knots_i[-1], num + 2)[1:-1]
    duplicates = np.where(np.isin(kns, knots_i))
    for i in duplicates:
        s = helpers.find_multiplicity(kns[i], knots_i)
        if s >= curve.degree:
            kns = np.delete(kns, i)

    # kns = np.delete(kns, duplicates)
    if len(kns) == 0:
        print("No new unique knots")
        return curve, changed

    rknot_curve = curve
    for k in kns:
        try:
            rknot_curve = Mop.insert_knot(rknot_curve, [k], [1])
        except GeomdlException:
            continue
    rknot_curve_err = get_error(profile_pts, rknot_curve)

    new_kv = rknot_curve.knotvector
    temp_kv = list(normalize(new_kv))
    try:
        rfit_curve = Mfitting.approximate_curve(list(map(tuple, profile_pts.values)), curve.degree, kv=temp_kv)
    except ValueError:
        rfit_curve = rknot_curve
        print("Cannot refit section with new kv")
    rfit_curve_err = get_error(profile_pts, rfit_curve)

    if np.average(rfit_curve_err) < np.average(rknot_curve_err) < np.average(e_i):
        rcrv = BSpline.Curve(normalize_kv=False)
        rcrv.degree = curve.degree
        rcrv.ctrlpts = rfit_curve.ctrlpts
        rcrv.knotvector = new_kv
        curve = rcrv
        changed = True
        print("Refit Section")
    elif np.average(rknot_curve_err) < np.average(e_i):
        curve = rknot_curve
        changed = True
        print("Insert Knot Section")

    # if changed:
    #     print("section changed")
    return curve, changed


def get_error(profile_pts, curve_obj, sep=False):
    """ Gets the fitting error for a single curve.

    :param profile_pts: dataframe of profile points
    :param curve_obj: fitted curve object
    :param sep: if True, returns a 2D array of x, y, z error values
    :return: fitted curve error, ndarray
    """
    import numpy as np  # pathos raises an 'np undefined' error without this
    points = profile_pts.values  # get array of points from df
    eval_idx = list(normalize(Mfitting.compute_params_curve(list(map(tuple, points))), low=min(curve_obj.knotvector), high=max(curve_obj.knotvector)))
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


def curve_plotting(profile_pts, crv, error_bound_value, med_filter=0, filter_plot=False, title="BSpline curve fit plot"):
    """ Plots a single curve in the X-Z plane with corresponding fitting error.

    :param profile_pts: profile data points
    :type profile_pts: pandas.DataFrame
    :param crv: curve to plot
    :type crv: BSpline.Curve
    :param error_bound_value: defined error bound for iterative fit as ratio of maximum curve value
    :type error_bound_value: float (must be between 0 and 1)
    :param med_filter: sets median filter window size if not None
    :type med_filter: float
    :param filter_plot: whether to plot the filtered curve
    :param title: title of the figure
    :type title: string
    :return: none
    """
    # font sizes
    # SMALL_SIZE = 16
    # MEDIUM_SIZE = 20
    # BIGGER_SIZE = 24
    LARGE_SIZE = 16

    # plt.rc('font', size=SMALL_SIZE)       # controls default text sizes
    # plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

    crv_pts = np.array(crv.evalpts)
    ct_pts = np.array(crv.ctrlpts)
    data_xz = profile_pts[['x', 'z']].values

    fig, ax = plt.subplots(2, figsize=(30, 24), sharex='all')
    if med_filter > 1:
        filtered_data = profile_pts[['x', 'y']]
        small_filter = int(np.sqrt(med_filter) + 1) if int(np.sqrt(med_filter)) >= 1 else 1
        filtered_z = nd.median_filter(nd.median_filter(nd.median_filter(profile_pts['z'].values,
                                                                        size=small_filter, mode='nearest'),
                                                       size=med_filter, mode='nearest'),
                                      size=small_filter, mode='nearest')
        filtered_data['z'] = filtered_z
        crv_err = get_error(filtered_data, crv, sep=True)

        # ax[0].plot(filtered_data['x'].values, filtered_data['z'].values, label='Median Filtered Data', c='purple', linewidth=2)

        if filter_plot:
            fig2, ax2 = plt.subplots(figsize=(24, 14))
            ax2.plot(data_xz[:, 0], data_xz[:, 1], label='Input Data', c='blue', linewidth=0.7)
            ax2.plot(filtered_data['x'].values, filtered_data['z'].values, label='Median Filtered Data', c='purple', linewidth=2)
            ax2.grid(True)
            ax2.legend(loc="upper right")
            ax2.set(xlabel='Lateral Position X [nm]', ylabel='Height Z [nm]')
            fig2.suptitle('Median Filter Result'.upper())
            # fig2.tight_layout()
            # plt.savefig("figures/med-fit.png")
    else:
        crv_err = get_error(profile_pts, crv, sep=True)

    ax[0].grid(True)
    ax[0].plot(data_xz[:, 0], data_xz[:, 1], label='Input Data', c='blue', linewidth=1.25, marker='.', markersize=1.5)
    ax[0].plot(crv_pts[:, 0], crv_pts[:, 2], label='Fitted Curve', c='red', linewidth=2)
    ax[0].plot(ct_pts[:, 0], ct_pts[:, 2], label='Control Points', marker='+', c='orange', linestyle='--', linewidth=0.75)

    scaled_kv = normalize(crv.knotvector, low=np.amin(profile_pts['x']), high=np.amax(profile_pts['x']))
    ax[0].hist(scaled_kv, bins=(int(len(profile_pts['x']) / 2)), bottom=(np.amin(profile_pts['z']) - 10), label='Knot Locations')

    ax[0].set_ylim(np.amin(profile_pts['z'].values) - 10, np.amax(profile_pts['z'].values) + 10)
    ax[0].set(xlabel='Lateral Position X [nm]', ylabel='Height Z [nm]',
              title='B-spline Result: Control Points={}, Data Size={},'.format(crv.ctrlpts_size, len(profile_pts)))

    ax[1].grid(True)
    ax[1].plot(data_xz[:, 0], crv_err[:, 0], 'k', label='Overall fitting error')

    # ax[1].plot(data_xz.values[:, 0], crv_err[:, 1], label='X error')
    # ax[1].plot(data_xz.values[:, 0], crv_err[:, 2], label='Y error')
    # ax[1].plot(data_xz.values[:, 0], crv_err[:, 3], label='Z error')

    ax[1].axhline(y=error_bound_value, xmin=data_xz[0, 0], xmax=data_xz[-1, 0], color='k', linestyle='--',
                  label='User-set error bound')
    ax[1].set(xlabel='Lateral Position X [nm]', ylabel='Error [nm]',
              title='Fitting Error: Max={}, Avg={}, Fitting Bound={} nm'.format(round(np.amax(crv_err), 4),
                                                                                round(np.average(crv_err), 4),
                                                                                round(error_bound_value, 2)))

    ax[0].legend(loc=(1.01, 0.5))
    ax[1].legend(loc=(1.01, 0.5))

    box = ax[0].get_position()
    ax[0].set_position([box.x0 - 0.065, box.y0, box.width, box.height])
    box = ax[1].get_position()
    ax[1].set_position([box.x0 - 0.065, box.y0, box.width, box.height])

    fig.suptitle(title.upper())
    # fig.tight_layout()
    fig_title = "figures/" + title.replace(' ', '-').lower() + "-cp{}".format(crv.ctrlpts_size) + ".png"
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

