import copy
from typing import Union
import os

import numpy as np
import pandas as pd
import pathos.multiprocessing as mp
from geomdl import helpers, BSpline, fitting
from geomdl.exceptions import GeomdlException
from matplotlib import pyplot as plt
from pathos.pools import ProcessPool as Pool
from scipy import ndimage as nd

from geomdl_mod import Mfitting, Moperations as Mop


# font sizes for matplotlib plots
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


def find_nearest(a0, a, mode='normal'):
    """
    Finds element in array `a` closest to the scalar value `a0`.
    Returns index position and value of result.
    Modes: ['less than', 'greater than', 'normal']
    """
    a = np.asarray(a)
    diff = a - a0
    if mode == 'greater than':
        diff[diff < 0] = np.inf
        idx = diff.argmin()
    elif mode == 'less than':
        diff[diff > 0] = -np.inf
        idx = diff.argmax()
    else:
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
        id_u, u_i = find_nearest(high, u_k)
        section_curve = Mop.split_curve(curve, u_i)[0]
        section_pts = np.array_split(profile_pts, [id_u + 1])[0]
        section_uk = np.array_split(u_k, [id_u + 1])[0]
    elif high == 1 or high == curve.knotvector[-1]:
        id_u, u_i = find_nearest(low, u_k)
        section_curve = Mop.split_curve(curve, u_i)[-1]
        section_pts = np.array_split(profile_pts, [id_u])[-1]
        section_uk = np.array_split(u_k, [id_u])[-1]
    else:
        id_u0, u_i0 = find_nearest(low, u_k)
        id_u1, u_i1 = find_nearest(high, u_k)
        section_curve = Mop.split_curve(Mop.split_curve(curve, u_i0)[-1], u_i1)[0]
        section_pts = np.array_split(profile_pts, [id_u0, id_u1 + 1])[1]
        section_uk = np.array_split(u_k, [id_u0, id_u1 + 1])[1]

    section_pts = pd.DataFrame(section_pts, columns=['x', 'y', 'z'])
    return section_curve, section_pts, section_uk


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
        if len(kv_new) > 100:   # for large datasets, the knot deletions that occur during merging will blow up the fit erro
            delete = s[i] - p
        else:                   # for smaller datasets, it is preferred not to increase the multiplicity so much
            delete = s[i] - 2
        if delete > 0:
            merged_curve = Mop.remove_knot(merged_curve, [join_knots[i]], [delete])

    return merged_curve


def adding_knots(profile_pts, curve, num, error_bound_value, u_k, randomized=False):
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
    :param randomized: if True, uses secondary knot selection method
    :type randomized: bool
    :return: refined curve
    """
    from geomdl_mod import Mfitting
    import numpy as np  # pathos raises an 'undefined' error without this

    e_i = get_error(profile_pts, curve)
    ei_max = np.amax(e_i)

    if ei_max > error_bound_value:
        knots_i = curve.knotvector
        kns = []
        choices = []
        rng = np.random.default_rng()
        if len(knots_i) + num >= len(profile_pts):
            # if number is too large, reset it to within the limit
            num = (len(profile_pts) - len(knots_i)) if (len(profile_pts) - len(knots_i)) > 0 else 1
            randomized = True
        if num > 0:      # primary knot selection method
            if num > len(e_i) - 2:  # make sure we don't go outside the index bounds of the interior array (exclude endpoints)
                num = len(e_i) - 2
            choices += list(u_k[list(np.argsort(e_i[1:-1])[::-1][:num])])    # get "num" of highest error locations on curve to choose from (again, exclude end points of e_i)
            if randomized:
                # sometimes the algorithm can get stuck if it's at the knot limit for a section
                # because oftentimes a centered knot or highest error knot already exists with multiplicity = p
                # so we set up a random knot selection to try to reduce error and avoid infinite loops
                choices += list(np.linspace(knots_i[0], knots_i[-1], num + 2)[1:-1])
            kns += list(rng.choice(choices, num, replace=False))
            kns = list(set(kns))    # ensures no duplicates (with replace=False this shouldn't be possible, but it doesn't hurt to be redundant)
        for k in kns:
            s = helpers.find_multiplicity(k, knots_i)
            if s < curve.degree:
                continue
            else:   # if s >= curve.degree:
                kns.remove(k)

        new_kv = list(np.asarray(knots_i + kns).astype(float))
        new_kv.sort()
        if new_kv[-1] > knots_i[-1] or new_kv[0] < knots_i[0]:  # I think I've fixed this bug, but just in case
            raise ValueError("One of the new knots is out of bounds!")

        temp_kv = list(normalize(new_kv))
        try:
            rfit_curve = Mfitting.approximate_curve(list(map(tuple, profile_pts.values)), curve.degree, kv=temp_kv)
        except ValueError:
            print("Cannot refit section with new kv")
            return curve
        rfit_err = get_error(profile_pts, rfit_curve)

        if np.average(rfit_err) < np.average(e_i):
            rcrv = BSpline.Curve(normalize_kv=False)
            rcrv.degree = curve.degree
            rcrv.ctrlpts = rfit_curve.ctrlpts
            rcrv.knotvector = new_kv
            curve = rcrv
            ei_max = np.amax(rfit_err)
            # print("Refit Section")
            curve_plotting(profile_pts, rcrv, error_bound_value, title=f"Refit curve section (Secondary = {randomized})")

    return curve


def adding_knots2(profile_pts, curve, num, error_bound_value, u_k, randomized=False):
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
    :param randomized: if True, uses randomized knot selection method
    :type randomized: bool
    :return: refined curve
    """
    from geomdl_mod import Mfitting
    import numpy as np  # pathos raises an 'undefined' error without this

    e_i = get_error(profile_pts, curve)
    ei_max = np.amax(e_i)

    knot_idx = 0    # idx to use for new knot (from sorted error array)
    while ei_max > error_bound_value:
        knots_i = curve.knotvector
        kns = []
        choices = []
        rng = np.random.default_rng()
        if len(knots_i) + num >= len(profile_pts):
            # if number is too large, reset it to within the limit
            num = (len(profile_pts) - len(knots_i)) if (len(profile_pts) - len(knots_i)) > 0 else 1
            randomized = True
        if num > 0:      # primary knot selection method
            if num > len(e_i) - 2:  # make sure we don't go outside the index bounds of the interior array (exclude endpoints)
                num = len(e_i) - 2
            if randomized:
                # sometimes the algorithm can get stuck if it's at the knot limit for a section
                # because oftentimes a centered knot or highest error knot already exists with multiplicity = p
                # so we set up a random knot selection to try to reduce error and avoid infinite loops
                kns += list(np.linspace(knots_i[0], knots_i[-1], num + 2)[1:-1])
            else:
                choices += list(u_k[list(np.argsort(e_i[1:-1])[::-1])])    # get descending highest error locations on curve to choose from (again, exclude end points of e_i)
                kns += choices[knot_idx:num]
        for k in kns:
            s = helpers.find_multiplicity(k, knots_i)
            if s < curve.degree:
                continue
            else:   # if s >= curve.degree:
                kns.remove(k)

        new_kv = list(np.asarray(knots_i + kns).astype(float))
        new_kv.sort()
        if new_kv[-1] > knots_i[-1] or new_kv[0] < knots_i[0]:  # I think I've fixed this bug, but just in case
            raise ValueError("One of the new knots is out of bounds!")

        temp_kv = list(normalize(new_kv))
        try:
            rfit_curve = Mfitting.approximate_curve(list(map(tuple, profile_pts.values)), curve.degree, kv=temp_kv)
        except ValueError:
            print("Cannot refit section with new kv")
            return curve
        rfit_err = get_error(profile_pts, rfit_curve)

        if np.amax(rfit_err) < ei_max:      # check if new curve has smaller max error
            rcrv = BSpline.Curve(normalize_kv=False)
            rcrv.degree = curve.degree
            rcrv.ctrlpts = rfit_curve.ctrlpts
            rcrv.knotvector = new_kv
            curve = rcrv
            ei_max = np.amax(rfit_err)
            # print("Refit Section")
            # curve_plotting(profile_pts, rcrv, error_bound_value, title=f"Refit curve section (Secondary = {randomized})")

        knot_idx += 1
        if knot_idx + num > len(choices):   # testing this in console: *doesn't* throw an error?? just returns fewer values moving toward the end of the array
            break                           # keeping this here just in case though
    return curve


def get_error(profile_pts, curve_obj, sep=False):
    """ Gets the fitting error for a single curve.

    :param profile_pts: dataframe of profile points
    :param curve_obj: fitted curve object
    :param sep: if True, returns a 2D array of x, y, z error values
    :return: fitted curve error, ndarray
    """
    import numpy as np  # pathos raises an 'np undefined' error without this
    points = profile_pts.values  # get array of points from df

    eval_idx = Mfitting.compute_params_curve(list(map(tuple, points)))
    if list(np.asarray(curve_obj.knotvector)[[0, -1]]) != [0.0, 1.0]:
        eval_idx = normalize(eval_idx, low=min(curve_obj.knotvector), high=max(curve_obj.knotvector))
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


def curve_plotting(profile_pts, crv, error_bound_value, sep=False, med_filter=0, title="BSpline curve fit plot", save_to=""):
    """ Plots a single curve in the X-Z plane with corresponding fitting error.

    :param profile_pts: profile data points
    :type profile_pts: pandas.DataFrame
    :param crv: curve to plot
    :type crv: BSpline.Curve
    :param error_bound_value: defined error bound for iterative fit as ratio of maximum curve value
    :type error_bound_value: float (must be between 0 and 1)
    :param sep: if True, plot the x, y and z error in error plot
    :type sep: bool
    :param med_filter: sets median filter window size if not None
    :type med_filter: float
    :param title: title of the figure
    :type title: string
    :param save_to: file directory to save figures
    :return: none
    """

    crv.delta = 0.001
    crv.evaluate()
    crv_pts = np.array(crv.evalpts)
    ct_pts = np.array(crv.ctrlpts)
    data_xz = profile_pts[['x', 'z']].values

    if np.amax(data_xz[:, 0]) / (10 ** 6) >= 1:
        scaling_xy = 10 ** 6
        xy_units = 'mm'
    elif np.amax(data_xz[:, 0]) / (10 ** 3) >= 1:
        scaling_xy = 10 ** 3
        xy_units = 'um'
    else:
        scaling_xy = 1
        xy_units = 'nm'

    fig, ax = plt.subplots(2, figsize=(40, 20), sharex='all')
    if med_filter > 1:
        filtered_data = profile_pts[['x', 'y']]
        small_filter = int(np.sqrt(med_filter) + 1) if int(np.sqrt(med_filter)) >= 1 else 1
        filtered_z = nd.median_filter(nd.median_filter(nd.median_filter(profile_pts['z'].values,
                                                                        size=small_filter, mode='nearest'),
                                                       size=med_filter, mode='nearest'),
                                      size=small_filter, mode='nearest')
        filtered_data['z'] = filtered_z
        crv_err = get_error(filtered_data, crv, sep=sep)

    else:
        crv_err = get_error(profile_pts, crv, sep=sep)

    ax[0].grid(True)
    ax[0].plot((data_xz[:, 0] / scaling_xy), data_xz[:, 1], label='Input Data', c='blue', marker='.', linewidth=2)
    ax[0].plot((crv_pts[:, 0] / scaling_xy), crv_pts[:, 2], label='Fitted Curve', c='red', linewidth=2)
    ax[0].plot((ct_pts[:, 0] / scaling_xy), ct_pts[:, 2], label='Control Points', marker='+', c='orange', linestyle='--', linewidth=2, alpha=0.7)

    scaled_kv = normalize(crv.knotvector, low=np.amin((profile_pts['x'] / scaling_xy)), high=np.amax((profile_pts['x'] / scaling_xy)))
    bottom = np.amin(profile_pts['z']) - (np.amax(profile_pts['z']) - np.amin(profile_pts['z'])) * 0.3
    top = np.amax(profile_pts['z']) + (np.amax(profile_pts['z']) - np.amin(profile_pts['z'])) * 0.3
    ax[0].hist(scaled_kv, bins=(int(len(profile_pts['x']))), bottom=bottom, label='Knot Locations')
    ax[0].set_ylim(bottom, top)
    ax[0].set(ylabel='Height Z [nm]',
              title=(title.upper() + ': Control Points={}, Data Size={}'.format(crv.ctrlpts_size, len(profile_pts))))

    ax[1].grid(True)
    if sep:
        ax[1].plot((data_xz[:, 0] / scaling_xy), crv_err[:, 0], 'k', label='Fitting error', linewidth=1.75)
        ax[1].plot((data_xz[:, 0] / scaling_xy), crv_err[:, 1], label='X error', c='blue', alpha=0.5)
        # ax[1].plot((data_xz[:, 0] / scaling_xy), crv_err[:, 2], label='Y error', c='green', alpha=0.3)
        ax[1].plot((data_xz[:, 0] / scaling_xy), crv_err[:, 3], label='Z error', c='red', alpha=0.7)
    else:
        ax[1].plot((data_xz[:, 0] / scaling_xy), crv_err, 'k', label='Fitting error')

    ax[1].axhline(y=error_bound_value, xmin=data_xz[0, 0], xmax=data_xz[-1, 0], color='k', linestyle='--',
                  label='Error bound', linewidth=1.75)
    ax[1].set(xlabel='Lateral Position X [{}]'.format(xy_units), ylabel='Error [nm]',
              title='Fitting Error: Max={}, Avg={}, Fitting Bound={} nm'.format(round(np.amax(crv_err), 4),
                                                                                round(np.average(crv_err), 4),
                                                                                round(error_bound_value, 2)))

    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    fig.tight_layout()

    if save_to != "":
        fig_title = title.replace(' ', '-').lower() + "-cp{}".format(crv.ctrlpts_size) + ".png"
        plt.savefig(os.path.join(save_to, fig_title))

    plt.show()


def plot_data_only(profile_pts, f_window=0, title="Raw Input Data", save_to=""):
    data_xz = profile_pts[['x', 'z']].values

    if np.amax(data_xz[:, 0]) / (10 ** 6) >= 1:
        scaling_xy = 10 ** 6
        xy_units = 'mm'
    elif np.amax(data_xz[:, 0]) / (10 ** 3) >= 1:
        scaling_xy = 10 ** 3
        xy_units = 'um'
    else:
        scaling_xy = 1
        xy_units = 'nm'

    fig, ax = plt.subplots(figsize=(40, 15))
    ax.grid(True)
    ax.plot((data_xz[:, 0] / scaling_xy), data_xz[:, 1], label='Input Data', c='blue', alpha=0.7)
    axtitle = title + " (points={})".format(len(profile_pts))

    if f_window > 1:
        small_filter = int(np.sqrt(f_window) + 1) if int(np.sqrt(f_window)) >= 1 else 1
        filtered_z = nd.median_filter(nd.median_filter(nd.median_filter(profile_pts['z'].values,
                                                                        size=small_filter, mode='nearest'),
                                                       size=f_window, mode='nearest'),
                                      size=small_filter, mode='nearest')
        axtitle += " & Median Filtered (window={})".format(f_window)
    else:
        filtered_z = None

    if filtered_z is not None:
        ax.plot((data_xz[:, 0] / scaling_xy), filtered_z, label='Median Filtered Data', c='purple', linewidth=3)
    ax.set(xlabel='Lateral Position X [{}]'.format(xy_units), ylabel='Height Z [nm]', title=axtitle)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save_to != "":
        fig_title = title.replace(' ', '-').lower() + ".png"
        save_path = os.path.join(save_to, fig_title)
        plt.savefig(save_path)

    plt.show()


def plot_curve3d(profile_pts, curve, title="3D Curve Plot"):
    """
    Plots the curve of interest in a 3D profile.

    :param profile_pts: The 3D profile points
    :type profile_pts: pd.DataFrame
    :param curve: The 3D curve
    :param title: title string
    :param save_to: directory path to save figure
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


def surf_plot(curves, title="Surface plotted from curve points"):
    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes(projection='3d')

    if np.amax(np.asarray(curves[0].evalpts).squeeze()[:, 0]) / (10 ** 6) >= 1:
        scaling_xy = 10 ** 6
        xy_units = 'mm'
    elif np.amax(np.asarray(curves[0].evalpts).squeeze()[:, 0]) / (10 ** 3) >= 1:
        scaling_xy = 10 ** 3
        xy_units = 'um'
    else:
        scaling_xy = 1
        xy_units = 'nm'

    stack_list = []
    for c in curves:
        c.delta = 0.001
        cv_pts = np.array(c.evalpts).squeeze()
        stack_list.append(cv_pts)
        ax.plot((cv_pts[:, 0] / scaling_xy), (cv_pts[:, 1] / scaling_xy), cv_pts[:, 2], 'ko')
    surf_pts = np.stack(stack_list, axis=2)

    ax.plot_surface((surf_pts[:, 0, :].T / scaling_xy), (surf_pts[:, 1, :].T / scaling_xy), surf_pts[:, 2, :].T,
                    alpha=0.5, cmap='viridis')
    ax.set(xlabel='Lateral Position X [{}]'.format(xy_units), ylabel='Lateral Position Y [{}]'.format(xy_units), zlabel='Height Z [nm]', title=title)
    ax.xaxis.labelpad = 45
    ax.yaxis.labelpad = 45
    ax.zaxis.labelpad = 45
    plt.show()
