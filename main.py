import numpy as np
import pandas as pd
from time import sleep
import pathos.multiprocessing as mp
from pathos.pools import ProcessPool as Pool
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from geomdl_mod import Mfitting, Moperations as Mop
from geomdl import operations as op, abstract, helpers, compatibility, NURBS
from geomdl.exceptions import GeomdlException


def find_nearest(a, a0, ret_val=True):
    """
    Finds element array `a` closest to the scalar value `a0`.
    Returns index position (and value) of result.
    """
    a = np.asarray(a)
    idx = np.abs(a - a0).argmin()
    value = a[idx]
    if ret_val:
        return idx, value
    else:
        return idx


def normalize(arr, low=0., high=1.):
    unit_norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    result = (unit_norm * (high - low)) + low
    return result


def split_knot_refinement(obj, dense=(1, 1, 1)):
    """ Similar to knot refinement, but returns recalculated knot vector and control points instead of the spline object.
        - authored emn 2022

    :param obj: spline geometry to be refined
    :type obj: abstract.SplineGeometry
    :param dense: refinement density in [u, v, w] format
    :type dense: list, tuple
    :return: new control points, new knot vector
    """
    new_cpts = new_kv = []
    # Start curve knot refinement
    if isinstance(obj, abstract.Curve):
        if dense[0] > 0:  # param is the refinement density in the form [u, v, w] = [#, #, #]
            cpts = obj.ctrlptsw if obj.rational else obj.ctrlpts
            new_cpts, new_kv = helpers.knot_refinement(obj.degree, obj.knotvector, cpts, density=dense[0])
            # obj.set_ctrlpts(new_cpts)
            # obj.knotvector = new_kv

    return new_cpts, new_kv


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


def iter_gen_curve(profile_pts, degree=3, cp_size_start=50, max_error=0.3, break_lim=10):
    """ Iterative curve fit for a single profile, increasing ctrlpts size each time based on max error limit.

    :param profile_pts: pandas dataframe of profile data points.
    :param degree: degree of fitted curve, default=3
    :param cp_size_start: number of control points to start with, default=10
    :param max_error: maximum error bound, as a percentage of the maximum Z value in the data
    :param break_lim: number of times to keep looping while there's no improvement in error (to prevent infinite loops if error bound is too small)
    :return: fitted curve object
    """
    from geomdl_mod import Mfitting  # pathos raises an 'MFitting undefined' error without this
    fit_pts_arr = profile_pts.values
    max_z = np.amax(fit_pts_arr[:, -1])
    error_bound = max_error * max_z

    fit_pts = list(map(tuple, fit_pts_arr))
    curve = Mfitting.approximate_curve(fit_pts, degree, ctrlpts_size=cp_size_start)

    fit_error = get_error(profile_pts, curve)
    cp_size = cp_size_start
    loops = 0
    while np.amax(fit_error) > error_bound and loops < break_lim:
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
            # curve_plotting(profile_pts, curve, fit_error)     # for debugging
            loops = 0
        else:
            loops += 1

    curve.delta = 0.001
    return curve


def iter_gen_curve2(profile_pts, degree=3, cp_size_start=50, max_error=0.3):
    """ Iterative curve fit for a single profile, adding knots each time based on max error limit.
        Uses op.insert_knot to generate refined curve.

    :param profile_pts: pandas dataframe of profile data points.
    :param degree: degree of fitted curve, default=3
    :param cp_size_start: number of control points to start with
    :param max_error: maximum error bound, as a ratio of the maximum Z value in the data
    :return: fitted curve object
    """
    fit_pts_arr = profile_pts.values
    max_z = np.amax(fit_pts_arr[:, -1])
    error_bound_value = max_error * max_z

    fit_pts = list(map(tuple, fit_pts_arr))
    curve = Mfitting.approximate_curve(fit_pts, degree, ctrlpts_size=cp_size_start)
    u_k = Mfitting.compute_params_curve(fit_pts)    # get u_k value conversions

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
    loops = 0   # prevents infinite loops
    while np.amax(fit_error) > error_bound_value:
        knots = np.array(curve.knotvector)
        interior_knots = knots[degree:-degree]  # multiplicity k = p + 1 for each end knot
        err_split = np.array_split(fit_error, splits)
        uk_split = np.array_split(u_k, splits)

        idx_knots_split = [(find_nearest(interior_knots, uk_split[si][-1], ret_val=False) + 1) for si in range(splits - 1)]

        knots_split = np.array_split(interior_knots, idx_knots_split)

        # new_knots = np.array([])
        # for si in range(splits):
        #     if np.amax(err_split[si]) > error_bound_value:
        #         kns = np.linspace(knots_split[si][0], knots_split[si][-1], add_knots)[1:-1]
        #         duplicates = np.where(np.isin(kns, knots_split[si]))
        #         kns = np.delete(kns, duplicates)
        #
        #         # TODO: how to check refit error by section instead of over entire rcurve?
        #         new_knots = np.concatenate((new_knots, kns))

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

            if np.average(rfit_error) < np.average(fit_error) or np.average(err_rcurve1) < np.average(fit_error):      # changed from max to average 7/26/22
                if np.average(rfit_error) < min(np.average(fit_error), np.average(err_rcurve1)):
                    curve = rcurve
                    fit_error = rfit_error
                    curve_plotting(profile_pts, curve, max_error, title='Iter refitting')  # for debugging
                elif np.average(err_rcurve1) < min(np.average(fit_error), np.average(rfit_error)):
                    curve = rcurve1
                    fit_error = err_rcurve1
                    curve_plotting(profile_pts, curve, max_error, title='Iter knot insertion')  # for debugging

        else:
            loops += 1  # no new knots
            if loops >= 3:  # no new knots over three loops
                break

        if splits <= (int(len(interior_knots / 2)) - add_splits):
            splits += add_splits  # only add splits up to half the number of data points
        else:
            add_knots += 1

    return curve


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


def curve_plotting(profile_pts, crv, max_error, title="NURBS curve fit plot"):
    """ Plots a single curve in the X-Z plane with corresponding fitting error.

    :param profile_pts: profile data points
    :type profile_pts: pandas.DataFrame
    :param crv: curve to plot
    :type crv: NURBS.Curve
    :param max_error: defined error bound for iterative fit as ratio of maximum curve value
    :type max_error: float (must be between 0 and 1)
    :param title: title of the figure
    :type title: string
    :return: none
    """

    crv_err = get_error(profile_pts, crv)
    error_bound_value = max_error * np.amax(profile_pts.values[:, -1])  # get physical value of error bound

    data_xz = profile_pts[['x', 'z']]
    crv_pts = np.array(crv.evalpts)
    ct_pts = np.array(crv.ctrlpts)

    fig, ax = plt.subplots(2, figsize=(20, 15), sharex='all')
    ax[0].grid(True)
    ax[0].plot(data_xz.values[:, 0], data_xz.values[:, 1], label='Input Data', c='blue', linewidth=1.5, marker='.', markersize=4)
    ax[0].plot(crv_pts[:, 0], crv_pts[:, 2], label='Fitted Curve', c='red', alpha=0.7)
    ax[0].plot(ct_pts[:, 0], ct_pts[:, 2], label='Control Points', marker='+', c='orange', linestyle='--', linewidth=0.75)
    ax[0].legend()
    ax[0].set(xlabel='Lateral Position X [nm]', ylabel='Height [nm]', title='B-spline Result: CP_size={}'.format(crv.ctrlpts_size))

    ax[1].grid(True)
    ax[1].plot(data_xz.values[:, 0], crv_err, 'k', label='Fitting error')
    ax[1].axhline(y=error_bound_value, xmin=data_xz.values[0, 0], xmax=data_xz.values[-1, 0], color='k', linestyle='--', label='User-set error bound')
    ax[1].set(xlabel='Lateral Position X [nm]', ylabel='Error [nm]', title='Fitting Error: Max={}, Avg={}'.format(round(np.amax(crv_err), 4), round(np.average(crv_err), 4)))
    ax[1].legend()

    fig.suptitle(title, fontsize=24)
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    filename = "lines_patt.csv"
    data_2d = pd.read_csv(filename, delimiter=',', names=['x', 'y', 'z'])
    profiles = len(set(data_2d['y'].values))
    deg = 3
    cpts_size = 50

    arr_splitting = np.array_split(data_2d, profiles)
    i = np.random.randint(0, len(arr_splitting))

    # curves_fit = parallel_fitting(arr_splitting, deg, cpts_size)
    # c = curves_fit[i]
    profile_df = arr_splitting[i]
    max_err = 0.2

    # c = iter_gen_curve(profile_df, max_error=max_err)
    c2 = iter_gen_curve2(profile_df, max_error=max_err)

    # curve_plotting(profile_df, c, max_err, title="Iteratively increased num control points")
    curve_plotting(profile_df, c2, max_err, title="Iteratively added knots in high error sections")
