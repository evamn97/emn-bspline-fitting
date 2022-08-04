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


def pbs_iter_curvefit(profile_pts, degree=3, cp_size_start=50, max_error=0.2):
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
    splits = mp.cpu_count()
    # add_splits = 1  # splits to add each iteration
    u_i = list(map(tuple, np.repeat(np.linspace(0, 1, splits + 1), 2)[1:-1].reshape((-1, 2))))  # get initial split locations

    unchanged_loops = 0
    final = False
    pool = Pool(mp.cpu_count())
    while final is False:
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

        # debugging
        if curve.ctrlpts_size == cp_size_start:
            curve_plotting(profile_pts.head(len(profiles_split[0])), curves_split[0], error_bound_value, uk=uk_split[0], med_filter=filter_size, filter_plot=True, title="Initial Fit")
            pass

        results1 = pool.amap(adding_knots, profiles_split, curves_split, [add_knots] * splits, [error_bound_value] * splits, uk_split)
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
        # print("Max error for sections = {}\n".format(max(section_err)))

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
        # print("Max error for rf_curve = {}".format(np.amax(rcurve_err)))
        # print("Max error for og_curve = {}\n".format(np.amax(fit_error)))

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

        if unchanged_loops > 6 and not all(section_err > error_bound_value):
            add_knots += 1
        # if unchanged_loops >= 10:
        #     print("Breaking out of loop after {} unchanged loops".format(unchanged_loops))
        #     break

    return curve


if __name__ == '__main__':
    start_time = dt.now()

    filename = "data/lines_patt3.csv"
    data_2d = pd.read_csv(filename, delimiter=',', names=['x', 'y', 'z'])
    profiles = len(set(data_2d['y'].values))
    deg = 3
    cpts_size = 1000

    arr_splitting = np.array_split(data_2d, profiles)
    # i = np.random.randint(0, len(arr_splitting))

    # curves_fit = parallel_fitting(arr_splitting, deg, cpts_size)
    # c = curves_fit[i]
    profile_df = arr_splitting[0]
    max_err = 0.07
    filter_window = 40
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
