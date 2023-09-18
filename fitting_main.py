from datetime import datetime as dt, timedelta
from time import sleep
import os
import numpy as np
from utils import *


def pbs_iter_curvefit(profile_pts, degree=3, cp_size_start=80, max_error=3., filter_size=30, save_to=""):
    """
    Iterative curve fit for a single profile, adding knots each time based on max error limit
    Uses parallel splitting based on the number of computer cpus.
    Includes median filter for noisy data.

    :param profile_pts: pandas dataframe of profile data points
    :param degree: degree of fitted curve, default=3
    :param cp_size_start: number of control points to start with
    :param max_error: maximum error bound, in nanometers
    :param filter_size: rolling window size for the median filter
    :param save_to: directory path to save results
    :return: fitted curve object
    """
    pbs_start = dt.now()

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

    max_cp_size = max((len(fit_pts) - degree - 11), int((len(fit_pts) / 10)))  # ensures greatest # knots is len(fit_pts) - 10
    assert (cp_size_start < max_cp_size), "cp_size_start must be smaller than maximum number of control points. \nGot cp_size_start={}, max_cp_size={}.".format(cp_size_start, max_cp_size)

    curve = Mfitting.approximate_curve(fit_pts, degree, ctrlpts_size=cp_size_start)
    fit_error = get_error(filtered_profile_pts, curve)

    print(f"{dt.now() - pbs_start} initial fit\ninitial max error: {np.amax(fit_error)}")

    curve_plotting(profile_pts, curve, max_error, med_filter=filter_size, title='Initial Curve Fit', save_to=save_to)  # for debugging
    u_k = Mfitting.compute_params_curve(fit_pts)  # get u_k value conversions

    add_knots = 1
    # too many splits will mess up the fit because of the knot deletion
    # that occurs during curve section merging
    splits = mp.cpu_count() if mp.cpu_count() < 13 else 13

    unchanged_loops = 0
    final = False
    randomized = False  # randomized knot selection method
    pool = Pool(mp.cpu_count())

    loop_times = []

    while final is False:

        loop_start = dt.now()

        u_i = list(map(tuple, np.repeat(np.linspace(0, 1, splits + 1), 2)[1:-1].reshape((-1, 2))))  # get initial split locations
        if np.amax(fit_error) < max_error:
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

        results1 = pool.amap(adding_knots2, profiles_split, curves_split,
                             [add_knots] * splits, [max_error] * splits,
                             uk_split, [randomized] * splits)
        while not results1.ready():
            sleep(1)
        rcurves_list = results1.get()

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

        if np.amax(rcurve_err) < np.amax(fit_error):
            print(f"refined max error = {np.amax(rcurve_err)}")
            curve = rcurve
            fit_error = rcurve_err
            unchanged_loops = 0
            # curve_plotting(filtered_profile_pts, rcurve, max_error, title="Refit Curve Plot")
        else:
            unchanged_loops += 1

        if unchanged_loops > 0:
            add_knots += 1
            print("no change")
        # elif unchanged_loops > 5:
        #     add_knots = 1
        #     # sometimes the alg gets stuck with the regular knot generation so
        #     # this will reset it to the randomized knot selection method
        #     randomized = True
        if unchanged_loops == 0:
            randomized = False
            add_knots = 1
            print("\nnew curve!\n")

        # print("add knots = {}".format(add_knots))
        # print("Max error for sections = {}\nMax error for rf_curve = {}\nMax error for og_curve = {}\n".format(np.amax(section_err),
        #                                                                                                        np.amax(rcurve_err),
        #                                                                                                        np.amax(fit_error)))

        loop_times.append(dt.now() - loop_start)

        print(f"Params for next loop: randomized = {randomized}, \t add_knots = {add_knots}")
        # pause = input("End of loop. Press any key to continue:\n")

    print(f"average loop time: {sum(loop_times, timedelta(0)) / len(loop_times)}\nfinal max error: {np.amax(fit_error)}\ncurve fit time: {dt.now() - pbs_start}\n")

    return curve


if __name__ == '__main__':
    start_time = dt.now()

    filename = "data/lines_patt1000spaced.csv"
    data_2d = pd.read_csv(filename, delimiter=',', names=['x', 'y', 'z'])
    profiles = len(set(data_2d['y'].values))
    arr_splitting = np.array_split(data_2d, profiles)

    deg = 3
    cpts_size = 30
    max_err = 2.0  # nanometers
    filter_window = 0
    save_dir = ""     # "cleanfit-results"

    profile_df = arr_splitting[0]
    # plot_data_only(profile_df, filter_window, save_to=save_dir)

    # single profile only
    curve = pbs_iter_curvefit(profile_df, cp_size_start=cpts_size, max_error=max_err, filter_size=filter_window)
    curve_plotting(profile_df, curve, max_err, med_filter=filter_window, title="Final BSpline Fit", save_to=save_dir)

    # putting the curves together
    # curves_u = []
    # for profile_df in arr_splitting:
    #     # profile_df *= (10 ** 9)   # if data is in meters
    #     c = pbs_iter_curvefit(profile_df, cp_size_start=cpts_size, max_error=max_err, filter_size=filter_window)
    #     curve_plotting(profile_df, c, max_err, med_filter=filter_window, title="Final BSpline Fit", save_to=save_dir)
    #     curves_u.append(c)
    # if len(curves_u) > 1:
    #     surf_plot(curves_u)

    end_time = dt.now()
    runtime = end_time - start_time
    print("Runtime: ", runtime)
