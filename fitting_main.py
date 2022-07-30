import numpy as np
import pandas as pd
from time import sleep
from typing import Union
import pathos.multiprocessing as mp
from pathos.pools import ProcessPool as Pool
from matplotlib import pyplot as plt
from scipy import ndimage as nd
from geomdl_mod import Mfitting, Moperations as Mop
from geomdl import operations as op, abstract, helpers, compatibility, NURBS
from geomdl.exceptions import GeomdlException
from datetime import datetime as dt


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


def gen_curve(profile_pts, degree, cp_size):  # generate a curve fit with fixed num of ctpts
    from geomdl_mod import Mfitting  # pathos raises an 'MFitting undefined' error without this
    fit_pts_arr = profile_pts.values
    fit_pts = list(map(tuple, fit_pts_arr))
    curve = Mfitting.approximate_curve(fit_pts, degree, ctrlpts_size=cp_size)
    curve.delta = 0.001
    return curve


def parallel_fitting(arr_split, deg, cp_size):
    in_deg = np.repeat(deg, len(arr_split))
    in_cp = np.repeat(cp_size, len(arr_split))
    pool = Pool(mp.cpu_count())
    curves_out = pool.map(gen_curve, arr_split, in_deg, in_cp)
    return curves_out


def iter_gen_curve(profile_pts, degree=3, cp_size_start=80, max_error=0.3):
    """ Iterative curve fit for a single profile, increasing ctrlpts size each time based on max error limit.

    :param profile_pts: pandas dataframe of profile data points.
    :param degree: degree of fitted curve, default=3
    :param cp_size_start: num of control points to start with
    :param max_error: maximum error bound, as a percentage of the maximum Z value in the data
    :return: fitted curve object
    """
    from geomdl_mod import Mfitting  # pathos raises an 'MFitting undefined' error without this

    fit_pts_arr = profile_pts.values
    max_z = np.amax(fit_pts_arr[:, -1])
    error_bound = max_error * max_z

    fit_pts = list(map(tuple, fit_pts_arr))

    max_cp_size = max((len(fit_pts) - degree - 11), int((len(fit_pts) / 10)))  # ensures greatest # knots is len(fit_pts) - 10
    assert (cp_size_start < max_cp_size), "cp_size_start must be smaller than maximum number of control points. \nGot cp_size_start={}, max_cp_size={}.".format(cp_size_start, max_cp_size)

    curve = Mfitting.approximate_curve(fit_pts, degree, ctrlpts_size=cp_size_start)
    fit_error = get_error(profile_pts, curve)

    cp_size = cp_size_start
    loops = 0
    while np.amax(fit_error) > error_bound and cp_size < max_cp_size:
        cp_size += 10
        try:
            rcurve = Mfitting.approximate_curve(fit_pts, degree, ctrlpts_size=cp_size)
            rfit_error = get_error(profile_pts, rcurve)
        except ValueError:  # can come up if knots become too dense
            loops += 1
            continue
        if np.amax(rfit_error) < np.amax(fit_error):
            curve = rcurve
            fit_error = rfit_error
            # curve_plotting(profile_pts, curve, error_bound)  # for debugging
            loops = 0
        else:
            loops += 1

    curve.delta = 0.001
    return curve


def pbs_iter_curvefit(profile_pts, degree=3, cp_size_start=50, max_error=0.3):
    """ Iterative curve fit for a single profile, adding knots each time based on max error limit.

    :param profile_pts: pandas dataframe of profile data points.
    :param degree: degree of fitted curve, default=3
    :param cp_size_start: number of control points to start with
    :param max_error: maximum error bound, as a ratio of the maximum Z value in the data
    :return: fitted curve object
    """
    fit_pts_arr = profile_pts.values
    error_bound_value = max_error * np.amax(profile_pts.values[:, -1])  # get physical value of error bound

    fit_pts = list(map(tuple, fit_pts_arr))

    max_cp_size = max((len(fit_pts) - degree - 11), int((len(fit_pts) / 10)))  # ensures greatest # knots is len(fit_pts) - 10
    assert (cp_size_start < max_cp_size), "cp_size_start must be smaller than maximum number of control points. \nGot cp_size_start={}, max_cp_size={}.".format(cp_size_start, max_cp_size)

    curve = Mfitting.approximate_curve(fit_pts, degree, ctrlpts_size=cp_size_start)
    u_k = Mfitting.compute_params_curve(fit_pts)  # get u_k value conversions

    fit_error = get_error(profile_pts, curve)
    add_knots = 3
    splits = 4  # splits to start with
    add_splits = 1  # splits to add each loop (if possible)

    def si_gen_knots(e_i, knots_i, i_s):
        """
        Generates new section knots if max(e_i) > error_bound_value, without increasing multiplicity of existing knots.
        :param e_i: section error array
        :type e_i: np.ndarray
        :param knots_i: section knots array
        :type knots_i: np.ndarray
        :param i_s: section index
        :type i_s: int
        """
        kns = []
        if np.amax(e_i) > error_bound_value:
            kns = np.linspace(knots_i[0], knots_i[-1], add_knots)[1:-1]
            duplicates = np.where(np.isin(kns, knots_i))
            kns = np.delete(kns, duplicates)

            rcrv = curve
            for k in kns:
                rcrv = Mop.insert_knot(rcrv, [k], [1])
            rcrv_err = np.array_split(get_error(profile_pts, rcrv), splits)[i_s]
            if np.average(rcrv_err) >= np.average(e_i):
                kns = []
        return list(kns)

    pool = Pool(mp.cpu_count())
    loops = 0  # prevents infinite loops
    while np.amax(fit_error) > error_bound_value:
        knots = np.array(curve.knotvector)
        interior_knots = knots[degree:-degree]  # multiplicity k = p + 1 for each end knot
        err_split = np.array_split(fit_error, splits)
        uk_split = np.array_split(u_k, splits)

        idx_knots_split = [(find_nearest(interior_knots, uk_split[si][-1])[0] + 1) for si in range(splits - 1)]

        knots_split = np.array_split(interior_knots, idx_knots_split)

        ids_s = np.arange(splits)
        results = pool.amap(si_gen_knots, err_split, knots_split, ids_s)
        while not results.ready():
            sleep(1)
        temp = results.get()
        new_knots = [k for kns in temp for k in kns if kns is not None]

        if len(new_knots) > 0:
            new_knots.sort()

            rcurve1 = curve
            for k in new_knots:
                rcurve1 = Mop.insert_knot(rcurve1, [k], [1])
            err_rcurve1 = get_error(profile_pts, rcurve1)

            ref_knots = list(np.concatenate((knots, new_knots)).flat)
            ref_knots.sort()

            if len(ref_knots) >= (len(fit_pts) - 10):  # no point in refitting if len(ref_knots) is getting close to len(fit_pts)
                break

            try:
                rcurve = Mfitting.approximate_curve(fit_pts, degree, kv=ref_knots)
                rcurve.delta = 0.001
                rcurve.evaluate()
                rfit_error = get_error(profile_pts, rcurve)
            except ValueError:  # if knots become too dense (I think that's the issue anyway, but this should hopefully be redundant)
                loops += 1
                continue

            if np.average(rfit_error) < np.average(fit_error) or np.average(err_rcurve1) < np.average(fit_error):  # changed from max to average 7/26/22
                if np.average(rfit_error) < min(np.average(fit_error), np.average(err_rcurve1)):
                    curve = rcurve
                    fit_error = rfit_error
                    curve_plotting(profile_pts, curve, error_bound_value, title='Iter refitting')  # for debugging
                elif np.average(err_rcurve1) < min(np.average(fit_error), np.average(rfit_error)):
                    curve = rcurve1
                    fit_error = err_rcurve1
                    curve_plotting(profile_pts, curve, error_bound_value, title='Iter knot insertion')  # for debugging

        else:
            loops += 1  # no new knots
            if loops >= 3:  # no new knots over three loops
                break

        if splits <= (int(len(interior_knots / 2)) - add_splits):
            splits += add_splits  # only add splits up to half the number of data points
        else:
            add_knots += 1

    return curve


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
    merged_curve = NURBS.Curve(normalize_kv=False)
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
    merged_curve = Mop.remove_knot(merged_curve, [join_knot], [s - p])

    return merged_curve


def adding_knots(profile_pts, curve, num, error_bound_value):
    """
    Adds num knots to the curve IF it reduces the fitting error.
    :param profile_pts: profile data points
    :type profile_pts: pandas.DataFrame
    :param curve: curve to plot
    :type curve: NURBS.Curve
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
        return curve, changed

    knots_i = curve.knotvector
    if len(knots_i) + num >= len(profile_pts):
        # print("Cannot add this many knots to curve without exceeding maximum knots limit.")
        return curve, changed

    kns = np.linspace(knots_i[0], knots_i[-1], num)[1:-1]
    duplicates = np.where(np.isin(kns, knots_i))
    kns = np.delete(kns, duplicates)
    if len(kns) == 0:
        return curve, changed

    rknot_curve = curve
    for k in kns:
        try:
            rknot_curve = Mop.insert_knot(rknot_curve, [k], [1])
        except GeomdlException:
            continue
    rknot_curve_err = get_error(profile_pts, rknot_curve)

    new_kv = sorted(np.concatenate((curve.knotvector, kns)))
    temp_kv = list(normalize(new_kv))
    try:
        rfit_curve = Mfitting.approximate_curve(list(map(tuple, profile_pts.values)), curve.degree, kv=temp_kv)
    except ValueError:
        rfit_curve = rknot_curve
    rfit_curve_err = get_error(profile_pts, rfit_curve)

    if np.average(rfit_curve_err) < np.average(rknot_curve_err) < np.average(e_i):
        rcrv = NURBS.Curve(normalize_kv=False)
        rcrv.degree = curve.degree
        rcrv.ctrlpts = rfit_curve.ctrlpts
        rcrv.knotvector = new_kv
        curve = rcrv
        changed = True
    elif np.average(rknot_curve_err) <= np.average(rfit_curve_err) < np.average(e_i):
        curve = rknot_curve
        changed = True

    return curve, changed


def pbs_iter_curvefit2(profile_pts, degree=3, cp_size_start=80, max_error=0.3, filter_size=30):
    """
    Iterative curve fit for a single profile, adding knots each time based on max error limit
    Uses parallel splitting based on the number of computer cpus.
    Includes median filter for noisy data.

    :param profile_pts: pandas dataframe of profile data points
    :param degree: degree of fitted curve, default=3
    :param cp_size_start: number of control points to start with
    :param max_error: maximum error bound, as a ratio of the maximum Z value in the data
    :param filter_size: rolling window size for the median filter
    :return: fitted curve object
    """
    if filter_size > 1:
        small_filter = int(np.sqrt(filter_size) + 1) if int(np.sqrt(filter_size)) >= 1 else 1
        filtered_z = nd.median_filter(nd.median_filter(nd.median_filter(profile_pts['z'].values,
                                                                        size=small_filter, mode='nearest'),
                                                       size=filter_size, mode='nearest'),
                                      size=small_filter, mode='nearest')
    else:
        filtered_z = profile_pts['z'].values
    filtered_profile_pts = pd.DataFrame({'x': profile_pts['x'].values, 'y': profile_pts['y'].values, 'z': filtered_z})
    fit_pts = list(map(tuple, filtered_profile_pts.values))
    error_bound_value = max_error * np.amax(filtered_profile_pts.values[:, -1])  # get physical value of error bound

    max_cp_size = max((len(fit_pts) - degree - 11), int((len(fit_pts) / 10)))  # ensures greatest # knots is len(fit_pts) - 10
    assert (cp_size_start < max_cp_size), "cp_size_start must be smaller than maximum number of control points. \nGot cp_size_start={}, max_cp_size={}.".format(cp_size_start, max_cp_size)

    curve = Mfitting.approximate_curve(fit_pts, degree, ctrlpts_size=cp_size_start)
    curve_plotting(profile_pts, curve, error_bound_value, med_filter=filter_size, filter_plot=True, title='Initial curve fit')  # for debugging
    u_k = Mfitting.compute_params_curve(fit_pts)  # get u_k value conversions

    fit_error = get_error(filtered_profile_pts, curve)
    add_knots = 3
    splits = mp.cpu_count() if mp.cpu_count() < 10 else 8  # splits to start with
    add_splits = 8  # splits to add each loop (if possible)

    unchanged_loops = 0
    pool = Pool(mp.cpu_count())
    while np.amax(fit_error) > error_bound_value:
        u_i = list(map(tuple, np.repeat(np.linspace(0, 1, splits + 1), 2)[1:-1].reshape((-1, 2))))
        results = pool.amap(section_curve, [curve] * splits, [filtered_profile_pts] * splits, u_i)
        while not results.ready():
            sleep(1)
        temp = results.get()
        curves_split = [c for (c, _) in temp]
        profiles_split = [p for (_, p) in temp]

        results1 = pool.amap(adding_knots, profiles_split, curves_split, [add_knots] * splits, [error_bound_value] * splits)
        while not results1.ready():
            sleep(1)
        temp = results1.get()
        rcurves_list = [r for (r, _) in temp]
        changed = [c for (_, c) in temp]

        rcurve = rcurves_list[0]
        for s in range(1, splits):
            rcurve = merge_curves(rcurve, rcurves_list[s])
        rcurve_err = get_error(filtered_profile_pts, rcurve)
        curve_plotting(profile_pts, rcurve, error_bound_value, med_filter=filter_size, title="Refined Curve")   # for debugging

        if np.average(rcurve_err) < np.average(fit_error):
            curve = rcurve
            fit_error = rcurve_err
            curve_plotting(profile_pts, curve, error_bound_value, med_filter=filter_size, title='Iter knot insert refit')  # for debugging

        if splits <= (int(len(fit_pts) / 4) - add_splits):
            splits += add_splits  # only add splits up to half the number of data points
        else:
            add_knots += 1
        if not any(c for c in changed):     # if none of the curve sections have changed, break
            unchanged_loops += 1

        # break conditions to avoid infinite loops
        if (len(curve.knotvector) + splits) >= (len(profile_pts) - 10):
            break
        if curve.ctrlpts_size >= max_cp_size:
            break
        if unchanged_loops > 10:
            break

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


def curve_plotting(profile_pts, crv, error_bound_value, med_filter: Union[None, float] = None, filter_plot=False, title="NURBS curve fit plot"):
    """ Plots a single curve in the X-Z plane with corresponding fitting error.

    :param profile_pts: profile data points
    :type profile_pts: pandas.DataFrame
    :param crv: curve to plot
    :type crv: NURBS.Curve
    :param error_bound_value: defined error bound for iterative fit as ratio of maximum curve value
    :type error_bound_value: float (must be between 0 and 1)
    :param med_filter: sets median filter window size if not None
    :type med_filter: Union[NoneType, float]
    :param filter_plot: whether to plot the filtered curve
    :param title: title of the figure
    :type title: string
    :return: none
    """
    # font sizes
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24
    LARGE_SIZE = 30

    plt.rc('font', size=SMALL_SIZE)             # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)       # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)       # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)       # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)       # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)      # legend fontsize
    plt.rc('figure', titlesize=LARGE_SIZE)      # fontsize of the figure title

    crv_pts = np.array(crv.evalpts)
    ct_pts = np.array(crv.ctrlpts)
    data_xz = profile_pts[['x', 'z']].values

    fig, ax = plt.subplots(2, figsize=(30, 18), sharex='all')
    if med_filter is not None:
        filtered_data = profile_pts[['x', 'y']]
        small_filter = int(np.sqrt(med_filter) + 1) if int(np.sqrt(med_filter)) >= 1 else 1
        filtered_z = nd.median_filter(nd.median_filter(nd.median_filter(profile_pts['z'].values,
                                                                        size=small_filter, mode='nearest'),
                                                       size=med_filter, mode='nearest'),
                                      size=small_filter, mode='nearest')
        filtered_data['z'] = filtered_z
        crv_err = get_error(filtered_data, crv, sep=True)

        ax[0].plot(filtered_data['x'].values, filtered_data['z'].values, label='Median Filtered Data', c='purple', linewidth=2)

        if filter_plot:
            fig2, ax2 = plt.subplots(figsize=(20, 10))
            ax2.plot(data_xz[:, 0], data_xz[:, 1], label='Input Data', c='blue', linewidth=0.7)
            ax2.plot(filtered_data['x'].values, filtered_data['z'].values, label='Median Filtered Data', c='purple', linewidth=2)
            ax2.grid(True)
            ax2.legend(loc="upper right")
            ax2.set(xlabel='Lateral Position X [nm]', ylabel='Height Z [nm]', title='Median Filter Result'.upper())
            fig2.tight_layout()
            # plt.savefig("figures/med-fit.png")
    else:
        crv_err = get_error(profile_pts, crv, sep=True)

    ax[0].grid(True)
    ax[0].plot(data_xz[:, 0], data_xz[:, 1], label='Input Data', c='blue', linewidth=1.5, marker='.', markersize=4)
    ax[0].plot(crv_pts[:, 0], crv_pts[:, 2], label='Fitted Curve', c='red', linewidth=2)
    ax[0].plot(ct_pts[:, 0], ct_pts[:, 2], label='Control Points', marker='+', c='orange', linestyle='--', linewidth=0.75)

    scaled_kv = normalize(crv.knotvector, low=np.amin(profile_pts['x']), high=np.amax(profile_pts['x']))
    ax[0].hist(scaled_kv, bins=(int(len(profile_pts['x']) / 2)), bottom=(np.amin(profile_pts['z']) - 10), label='Knot Locations')

    ax[0].legend(loc="upper right")
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
    ax[1].legend(loc="upper right")

    fig.suptitle(title.upper(), fontsize=30)
    fig.tight_layout()
    fig_title = "figures/" + title.replace(' ', '-').lower() + "-cp{}".format(crv.ctrlpts_size) + ".png"
    # plt.savefig(fig_title)

    plt.show()


def plot_curve3d(profile_pts, crv, title="3D Curve Plot"):
    """
    Plots the curve of interest in a 3D profile.

    :param profile_pts: The 3D profile points
    :type profile_pts: pd.DataFrame
    :param title: title string
    :return: None
    """



if __name__ == '__main__':
    start_time = dt.now()

    filename = "data/lines_patt3.csv"
    data_2d = pd.read_csv(filename, delimiter=',', names=['x', 'y', 'z'])
    profiles = len(set(data_2d['y'].values))
    deg = 3
    cpts_size = 50

    arr_splitting = np.array_split(data_2d, profiles)
    # i = np.random.randint(0, len(arr_splitting))

    # curves_fit = parallel_fitting(arr_splitting, deg, cpts_size)
    # c = curves_fit[i]
    profile_df = arr_splitting[0]
    max_err = 0.18
    filter_window = 30

    # c = iter_gen_curve(profile_df, max_error=max_err, cp_size_start=80)
    # curve_plotting(profile_df, c, max_err, title="Iteratively increased num control points")

    cv2 = pbs_iter_curvefit2(profile_df, cp_size_start=cpts_size, max_error=max_err, filter_size=filter_window)
    curve_plotting(profile_df, cv2, max_err, med_filter=filter_window, title="Final BSpline Fit")

    end_time = dt.now()
    runtime = end_time - start_time
    print("Runtime: ", runtime)
