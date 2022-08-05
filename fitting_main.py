from datetime import datetime as dt
from time import sleep
import numpy as np
from utils import *


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


def pbs_iter_curvefit2(profile_pts, degree=3, cp_size_start=80, max_error=0.2, filter_size=30):
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
    # curve_plotting(profile_pts, curve, error_bound_value, med_filter=filter_size, filter_plot=True, title='Initial Curve Fit')  # for debugging
    u_k = Mfitting.compute_params_curve(fit_pts)  # get u_k value conversions

    fit_error = get_error(filtered_profile_pts, curve)
    add_knots = 1
    # too many splits will mess up the fit because of the knot deletion
    # that occurs during curve section merging
    splits = mp.cpu_count() if mp.cpu_count() < 13 else 13

    unchanged_loops = 0
    final = False
    secondary = False   # secondary knot selection method
    pool = Pool(mp.cpu_count())
    while final is False:
        u_i = list(map(tuple, np.repeat(np.linspace(0, 1, splits + 1), 2)[1:-1].reshape((-1, 2))))  # get initial split locations
        if np.amax(fit_error) < error_bound_value:
            final = True
            add_knots = 0
            splits = np.ceil(mp.cpu_count() / 3).astype(int) if np.ceil(mp.cpu_count() / 3) >= 3 else 3
            u_i = list(map(tuple, np.repeat(np.linspace(0, 1, splits + 1), 2)[1:-1].reshape((-1, 2))))
        try:
            results = pool.amap(get_curve_section, [curve] * splits, [filtered_profile_pts] * splits, u_i)
            temp = results.get()
        except GeomdlException:
            raise GeomdlException("Cannot split the curve at these locations. Check the number of splits.")
        while not results.ready():
            sleep(1)

        curves_split = [c for (c, _, _) in temp]
        profiles_split = [p for (_, p, _) in temp]
        uk_split = [u for (_, _, u) in temp]

        results1 = pool.amap(adding_knots, profiles_split, curves_split,
                             [add_knots] * splits, [error_bound_value] * splits,
                             uk_split, [secondary] * splits)
        while not results1.ready():
            sleep(1)
        temp = results1.get()
        rcurves_list = [r for (r, _) in temp]
        changed = [c for (_, c) in temp]

        section_err = []
        for i in range(len(rcurves_list)):
            rc = rcurves_list[i]
            pr = profiles_split[i]
            err = np.amax(get_error(pr, rc))
            section_err.append(err)

        rcurve = merge_curves_multi(rcurves_list)
        rknots_set = sorted(list(set(rcurve.knotvector)))
        results2 = pool.amap(helpers.find_multiplicity, rknots_set, [rcurve.knotvector] * len(rcurve.knotvector))
        while not results2.ready():
            sleep(1)
        s = results2.get()
        delete = list(np.where(np.asarray(s) > degree)[0])[1:-1] + list(np.where(np.asarray(s) > degree + 1)[0])
        for d in delete:
            k = rknots_set[d]
            if k == 0.0 or k == 1.0:
                if s[d] > (degree + 1):
                    rcurve = Mop.remove_knot(rcurve, [k], [s[d] - degree - 1])
            else:
                rcurve = Mop.remove_knot(rcurve, [k], [s[d] - degree])

        rcurve_err = get_error(filtered_profile_pts, rcurve)

        if not any(c for c in changed):  # if none of the curve sections have changed
            unchanged_loops += 1

        if np.amax(rcurve_err) < np.amax(fit_error):
            # if np.amax(rcurve_err) > error_bound_value:
            #     curve_plotting(filtered_profile_pts, rcurve, error_bound_value, title="Refit Curve Plot")
            curve = rcurve
            fit_error = rcurve_err
            unchanged_loops = 0
        else:
            unchanged_loops += 1

        if unchanged_loops > 2:
            add_knots += 1
            # sometimes the alg gets stuck with the regular knot generation so
            # this will reset it to the secondary knot selection method, which is randomized
            secondary = True
        if unchanged_loops == 0:
            secondary = False

        # print("add knots = {}".format(add_knots))
        print("Max error for sections = {}\nMax error for rf_curve = {}\nMax error for og_curve = {}".format(np.amax(section_err),
                                                                                                             np.amax(rcurve_err),
                                                                                                             np.amax(fit_error)))
        # print("Unchanged loops = {}".format(unchanged_loops))

    return curve


if __name__ == '__main__':
    start_time = dt.now()

    filename = "data/lines_patt20000.csv"
    data_2d = pd.read_csv(filename, delimiter=',', names=['x', 'y', 'z'])
    profiles = len(set(data_2d['y'].values))
    deg = 3
    cpts_size = 250

    arr_splitting = np.array_split(data_2d, profiles)

    profile_df = arr_splitting[0]
    max_err = 0.08
    filter_window = 0
    if filter_window > 1:
        small_filter = int(np.sqrt(filter_window) + 1) if int(np.sqrt(filter_window)) >= 1 else 1
        filtered_z = nd.median_filter(nd.median_filter(nd.median_filter(profile_df['z'].values,
                                                                        size=small_filter, mode='nearest'),
                                                       size=filter_window, mode='nearest'),
                                      size=small_filter, mode='nearest')
        error_bound = max_err * np.amax(filtered_z)
    else:
        error_bound = max_err * np.amax(profile_df['z'].values)

    cv2 = pbs_iter_curvefit2(profile_df, cp_size_start=cpts_size, max_error=max_err, filter_size=filter_window)
    curve_plotting(profile_df, cv2, error_bound, med_filter=filter_window, title="Final BSpline Fit")

    end_time = dt.now()
    runtime = end_time - start_time
    print("Runtime: ", runtime)
