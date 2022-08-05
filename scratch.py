import numpy as np
from time import sleep

import pandas as pd

from utils import *


def get_section_multi(profile_pts: pd.DataFrame, splits: int, cp_size: int):
    """
        Returns a list of data and u_k sections defined by low and high u_k values. End points are overlapping
        :param profile_pts: data points the input curve is fitted to
        :param splits: how many sections to return
        :param cp_size: number of total control points (must be divisible by splits)
        :return: list of curve section (with clamped exterior knots), list of section data points
        """
    cp_size_section = int(cp_size / splits)
    degree = len(profile_pts.columns)

    profiles_split = np.array_split(profile_pts, splits)
    p_join = [p.values[-1] for p in profiles_split][:-1]

    u_k = Mfitting.compute_params_curve(list(map(tuple, profile_pts.values)))
    uk_split = np.array_split(u_k, splits)
    u_join = [u[-1] for u in uk_split][:-1]

    kv = Mfitting.compute_knot_vector2(degree, len(profile_pts), cp_size, u_k)
    kv_split = np.array_split(kv, splits)
    k_join = np.repeat([k[-1] for k in kv_split[:-1]], degree).reshape((-1, degree))

    for s in range(splits - 1):
        p0 = p_join[s]
        u0 = u_join[s]
        k0 = k_join[s]
        uk_split[s + 1] = np.append(uk_split[s + 1], u0)
        df_temp = pd.DataFrame(columns=['x', 'y', 'z'])
        df_temp.loc[0] = p0.tolist()
        profiles_split[s + 1] = pd.concat((profiles_split[s + 1],
                                           df_temp))
        kv_split[s] = np.concatenate((kv_split[s], k_join[s]))
        kv_split[s + 1] = np.concatenate((k_join[s], kv_split[s + 1]))

    return profiles_split, uk_split, kv_split


def section_fit(profile_pts, u_k, kv):
    """
    Fits a section of data points using a given  u_k parameterization, and knot vector
    :param profile_pts: profile data points
    :type profile_pts: pandas.DataFrame
    :param u_k: parametric coordinates of profile_pts
    :type u_k: numpy.ndarray
    :param kv: knot vector to use for fitting
    :return: refined curve
    """
    from geomdl_mod import Mfitting
    import numpy as np  # pathos raises an 'undefined' error without this

    degree = len(profile_pts.columns)
    kv.sort()
    temp_kv = list(normalize(kv))

    # TODO: how to do error handling for this? (ValueError when fitting fails)
    temp_curve = Mfitting.approximate_curve(list(map(tuple, profile_pts.values)),
                                            degree, kv=temp_kv)

    curve_section = BSpline.Curve(normalize_kv=False)
    curve_section.degree = degree
    curve_section.ctrlpts = temp_curve.ctrlpts
    curve_section.knotvector = list(kv)
    curve_section.delta = 0.001
    curve_section.evaluate()

    return curve_section


def merge_curves_multi2(args):
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
    kv_new = args[0].knotvector[:p+1]
    s = []

    for c in args:
        ctrlpts_new = ctrlpts_new + c.ctrlpts
        kv_new = sorted(kv_new + c.knotvector[p+1:])    # ensures rule m = n + p + 1
        join_knots.append(c.knotvector[-1])
    join_knots = join_knots[:-1]
    for j in join_knots:
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
        if len(kv_new) > 100:   # for large datasets, the knot deletions that occur during merging will blow up the fit error
            delete = s[i] - p
        else:                   # for smaller datasets, it is preferred not to increase the multiplicity so much
            delete = s[i] - 2
        if delete > 0:
            merged_curve = Mop.remove_knot(merged_curve, [join_knots[i]], [delete])

    return merged_curve


def pbs_iter_curvefit3(profile_pts, degree=3, cp_size_start=80, max_error=0.2):
    """
    Iterative curve fit for a single profile, adding knots each time based on max error limit
    Uses parallel splitting based on the number of computer cpus.
    Includes median filter for noisy data.

    :param profile_pts: pandas dataframe of profile data points
    :param degree: degree of fitted curve, default=3
    :param cp_size_start: number of control points to start with
    :param max_error: maximum error bound, as a ratio of the maximum Z value in the data
    :return: fitted curve object
    """
    fit_pts = list(map(tuple, profile_pts.values))
    error_bound_value = max_error * np.amax(profile_pts.values[:, -1])  # get physical value of error bound

    max_cp_size = max((len(fit_pts) - degree - 11), int((len(fit_pts) / 10)))  # ensures greatest # knots is len(fit_pts) - 10
    assert (cp_size_start < max_cp_size), "cp_size_start must be smaller than maximum number of control points. \nGot cp_size_start={}, max_cp_size={}.".format(cp_size_start, max_cp_size)
    # too many splits will mess up the fit because of the knot deletion
    # that occurs during curve section merging
    splits = mp.cpu_count() if mp.cpu_count() < 13 else 13
    pool = Pool(mp.cpu_count())
    profiles_split, uk_split, kv_split = get_section_multi(profile_pts, splits, cp_size_start)
    results = pool.amap(section_fit, profiles_split, uk_split, kv_split)
    while not results.ready():
        sleep(1)
    curves_split = results.get()
    curve = merge_curves_multi2(curves_split)
    fit_error = get_error(profile_pts, curve)
    add_knots = 0
    unchanged_loops = 0
    final = False
    secondary = False  # secondary knot selection method
    while final is False:
        if np.amax(fit_error) < error_bound_value:
            final = True
            add_knots = 0
            splits = np.ceil(mp.cpu_count() / 3).astype(int) if np.ceil(mp.cpu_count() / 3) >= 3 else 3
            u_i = list(map(tuple, np.repeat(np.linspace(0, 1, splits + 1), 2)[1:-1].reshape((-1, 2))))
            try:
                results = pool.amap(get_curve_section, [curve] * splits, [profile_pts] * splits, u_i)
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

        rcurve = merge_curves_multi2(rcurves_list)
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

        rcurve_err = get_error(profile_pts, rcurve)

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
        if add_knots == 0:
            add_knots = 1
        if unchanged_loops > 2:
            add_knots += 1
            # sometimes the alg gets stuck with the regular knot generation so
            # this will reset it to the secondary knot selection method, which is randomized
            secondary = True
        if unchanged_loops == 0:
            secondary = False

        print("add knots = {}".format(add_knots))
        print("Max error for sections = {}\nMax error for rf_curve = {}\nMax error for og_curve = {}".format(np.amax(section_err),
                                                                                                             np.amax(rcurve_err),
                                                                                                             np.amax(fit_error)))
        print("Unchanged loops = {}".format(unchanged_loops))

    return curve


if __name__ == '__main__':
    filename = "data/lines_patt20000.csv"
    data_2d = pd.read_csv(filename, delimiter=',', names=['x', 'y', 'z'])
    profiles = len(set(data_2d['y'].values))

    arr_splitting = np.array_split(data_2d, profiles)
    profile_df = arr_splitting[0]
    fitting_pts = profile_df[['x', 'y']]
    deg = 3
    cpts_size = 250
    max_err = 0.08
    filter_window = 0
    if filter_window > 1:
        s_win = int(np.sqrt(filter_window) + 1) if int(np.sqrt(filter_window)) >= 1 else 1
        medfilt_z = nd.median_filter(nd.median_filter(nd.median_filter(profile_df['z'].values,
                                                                       size=s_win, mode='nearest'),
                                                      size=filter_window, mode='nearest'),
                                     size=s_win, mode='nearest')
        error_bound = max_err * np.amax(medfilt_z)
        fitting_pts['z'] = medfilt_z
    else:
        error_bound = max_err * np.amax(profile_df['z'].values)
        fitting_pts['z'] = profile_df['z'].values

    cv3 = pbs_iter_curvefit3(fitting_pts, cp_size_start=cpts_size, max_error=max_err)
    curve_plotting(profile_df, cv3, error_bound, med_filter=filter_window, title="Final BSpline Fit")
