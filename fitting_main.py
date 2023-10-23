from datetime import datetime as dt, timedelta
from time import sleep
from geomdl.exceptions import GeomdlException
from warnings import warn
from argparse import ArgumentParser
from utils import *


def pbs_iter_curvefit(profile_pts, cp_size_start, max_error, filter_size, save_to="", degree=3):
    """
    Iterative curve fit for a single profile, adding knots each time based on max error limit
    Uses parallel splitting based on the number of computer cpus.
    Includes median filter for noisy data.

    :param profile_pts: pandas dataframe of profile data points
    :param cp_size_start: number of control points to start with
    :param max_error: maximum error bound, in nanometers
    :param filter_size: rolling window size for the median filter
    :param save_to: directory path to save results
    :param degree: degree of fitted curve, default=3
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
    plot_data_only(profile_pts, f_window=filter_size, save_to=save_to)     # plot data to be fitted

    max_cp_size = max((len(fit_pts) - degree - 11), int((len(fit_pts) / 10)))  # ensures greatest # knots is len(fit_pts) - 10
    assert (cp_size_start < max_cp_size), "cp_size_start must be smaller than maximum number of control points. \nGot cp_size_start={}, max_cp_size={}.".format(cp_size_start, max_cp_size)

    curve = Mfitting.approximate_curve(fit_pts, degree, ctrlpts_size=cp_size_start)
    fit_error = get_error(filtered_profile_pts, curve)

    print(f"{dt.now() - pbs_start} Initial fit time\nInitial curve max error = {round(np.amax(fit_error), 4)} nm")

    curve_plotting(profile_pts, curve, max_error, med_filter=filter_size, sep=False, title='Initial Curve Fit', save_to=save_to)     # plot initial curve fit
    u_k = Mfitting.compute_params_curve(fit_pts)  # get u_k value conversions

    # too many splits will mess up the fit because of the knot deletion
    # that occurs during curve section merging
    splits = mp.cpu_count() if mp.cpu_count() < 13 else 13
    pool = Pool(mp.cpu_count())

    while np.amax(fit_error) > max_error:
        # get split locations based on number of splits, and split curve & data
        u_i = list(map(tuple, np.repeat(np.linspace(0, 1, splits + 1), 2)[1:-1].reshape((-1, 2))))  # get initial split locations
        try:
            results = pool.amap(get_curve_section, [curve] * splits, [filtered_profile_pts] * splits, u_i)
        except GeomdlException:
            raise GeomdlException("Cannot split the curve at these locations. Check the number of splits.")
        while not results.ready():
            sleep(1)
        temp = results.get()

        curves_split = [c for (c, _, _) in temp]
        profiles_split = [p for (_, p, _) in temp]
        uk_split = [u for (_, _, u) in temp]

        # ******************** KNOT INSERTION ROUTINE ********************
        results1 = pool.amap(adding_knots, profiles_split, curves_split, [max_error] * splits, uk_split)
        while not results1.ready():
            sleep(1)
        rcurves_list = results1.get()
        # *****************************************************************

        # get error of refit curve sections
        section_err = []
        for i in range(len(rcurves_list)):
            rc = rcurves_list[i]
            pr = profiles_split[i]
            err = np.amax(get_error(pr, rc))
            section_err.append(err)

        # merge refit curve sections into new curve
        rcurve = merge_curves_multi(rcurves_list)

        # get multiplicity of all knots in new curve and delete any that violate s > p (except for endpoints where s = p + 1)
        # this should be redundant... (hopefully)
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

        # get new curve error & check against error bound
        rcurve_err = get_error(filtered_profile_pts, rcurve)
        if np.amax(rcurve_err) < max_error:
            curve = rcurve
            fit_error = rcurve_err
        else:
            cp_size_start += 10
            curve = Mfitting.approximate_curve(fit_pts, degree, ctrlpts_size=cp_size_start)
            fit_error = get_error(filtered_profile_pts, curve)
            if cp_size_start > max_cp_size:
                warn("\nCould not fit to given error bound. Try increasing max_error. \nReturning curve...")
                break
        # for debugging
        # pause = input("End of loop. Press any key to continue:\n")

    print(f"final max error: {np.amax(fit_error)} nm\ncurve fit time: {dt.now() - pbs_start}\n")
    curve_plotting(profile_pts, curve, max_error, med_filter=filter_size, sep=False, title='Final Curve Fit', save_to=save_to)
    return curve


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-f', '--filename', help='file path of input', type=str)
    parser.add_argument('-s', '--save-dir', type=str, help='saves figures if not empty', default="")
    parser.add_argument('-e', '--error-bound', type=float, help='fitting bound, in nanometers', default=1.0)
    parser.add_argument('-cp', '--ctpts-size', type=int, help='initial control points size for fitting', default=30)
    parser.add_argument('-fw', '--filter-window', type=int, help='filter window size for noisy data', default=0)

    return parser.parse_args()


def get_profiles(filename):
    data_xyz = pd.read_csv(filename, delimiter=',', names=['x', 'y', 'z'])
    num_profile = len(set(data_xyz['y'].values))
    profile_arrs = np.array_split(data_xyz, num_profile)

    return profile_arrs


if __name__ == '__main__':
    # start_time = dt.now()

    # params = parse_args()
    # file = params.filename
    # save_dir = params.save_dir
    # max_err = params.error_bound
    # ctpts_size = params.ctpts_size
    # filter_window = params.filter_window

    # debugging
    # file = "data_new/lines_5000_clean.csv"
    save_dir = "output_files"
    max_err = 1.0
    ctpts_size = 250
    filter_window = 0

    source = "data_new/time_trials"
    files = [file for file in sorted(os.listdir(source)) if '.csv' in file]
    for file in files:
        print(f'Fitting to {os.path.basename(file)}...')

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        profile_df = get_profiles(os.path.join(source, file))[0]
    # profile_df = get_profiles(file)[0]

        # see if it's saved in meters, convert to nm for consistency
        if profile_df['z'].max() <= 10**-6:
            profile_df *= 10 ** 9
            if profile_df['z'].min() < 0:
                profile_df['z'] -= profile_df['z'].min()

        if 'noisy' in file or 'experimental' in file:
            filter_window = 30
        else:
            filter_window = 0

        start_time = dt.now()
        # single profile only
        curve = pbs_iter_curvefit(profile_df, cp_size_start=ctpts_size, max_error=max_err, filter_size=filter_window, save_to=save_dir)
        runtime = dt.now() - start_time

        with open(f"{os.path.join(save_dir, os.path.splitext(os.path.basename(file))[0])}.log", 'w') as logfile:
            logfile.write(f'Filename: {file}\n'
                          f'Number of points: {len(profile_df)}\n'
                          f'Initial control points: {ctpts_size}\n'
                          f'Filter window: {filter_window}\n'
                          f'Final control points: {curve.ctrlpts_size}\n'
                          f'Runtime: {runtime}')

        print(f'Runtime: {runtime}\nDone with file {os.path.splitext(file)[0]}! \n')

    # putting the curves together
    # curves_u = []
    # for profile_df in arr_splitting:
    #     # profile_df *= (10 ** 9)   # if data is in meters
    #     c = pbs_iter_curvefit(profile_df, cp_size_start=cpts_size, max_error=max_err, filter_size=filter_window)
    #     curve_plotting(profile_df, c, max_err, med_filter=filter_window, title="Final BSpline Fit", save_to=save_dir)
    #     curves_u.append(c)
    # if len(curves_u) > 1:
    #     surf_plot(curves_u)

    # runtime = dt.now() - start_time
    # print("Runtime: ", runtime)
