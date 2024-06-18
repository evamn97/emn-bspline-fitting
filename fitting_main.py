from datetime import datetime as dt, timedelta
from time import sleep
from geomdl.exceptions import GeomdlException
from warnings import warn
from argparse import ArgumentParser
import copy
from utils import *


def pbs_iter_curvefit(profile_pts, cp_size_start, err_bound, med_filter, save_to="", degree=3):
    """
    Iterative curve fit for a single profile, adding knots each time based on max error limit
    Uses parallel splitting based on the number of computer cpus.
    Includes median filter for noisy data.

    :param profile_pts: pandas dataframe of profile data points
    :param cp_size_start: number of control points to start with
    :param err_bound: maximum error bound, in nanometers
    :param med_filter: rolling window size for the median filter
    :param save_to: directory path to save results
    :param degree: degree of fitted curve, default=3
    :return: fitted curve object
    """
    pbs_start = dt.now()

    if med_filter > 1:
        small_filter = int(np.sqrt(med_filter) + 1) if int(np.sqrt(med_filter)) >= 1 else 1
        filtered_z = nd.median_filter(nd.median_filter(nd.median_filter(profile_pts['z'].values,
                                                                        size=small_filter, mode='nearest'),
                                                       size=med_filter, mode='nearest'),
                                      size=small_filter, mode='nearest')
    else:
        filtered_z = profile_pts['z'].values
    filtered_profile_pts = pd.DataFrame({'x': profile_pts['x'].values, 'y': profile_pts['y'].values, 'z': filtered_z})
    fit_pts = list(map(tuple, filtered_profile_pts.values))
    plot_data_only(profile_pts, med_filter=med_filter, save_to=save_to, plot_show=True)  # plot data to be fitted

    max_cp_size = max((len(fit_pts) - degree - 11),
                      int((len(fit_pts) / 10)))  # ensures greatest # knots is len(fit_pts) - 10
    assert (
                cp_size_start < max_cp_size), "cp_size_start must be smaller than maximum number of control points. \nGot cp_size_start={}, max_cp_size={}.".format(
        cp_size_start, max_cp_size)

    curve = Mfitting.approximate_curve(fit_pts, degree, ctrlpts_size=cp_size_start)
    curve_0 = copy.deepcopy(curve)
    fit_error = get_error(filtered_profile_pts, curve)

    init_time = dt.now() - pbs_start
    print(f"{init_time} Initial fit time\nInitial fit max error = {round(np.amax(fit_error), 4)} nm")

    curve_plotting(profile_pts, curve, err_bound, med_filter=med_filter, sep=False, title='Initial Fit',
                   save_to=save_to, plot_show=True)  # plot initial curve fit
    u_k = Mfitting.compute_params_curve(fit_pts)  # get u_k value conversions

    # too many splits will mess up the fit because of the knot deletion
    # that occurs during curve section merging
    splits = mp.cpu_count() if mp.cpu_count() < 13 else 13
    pool = Pool(mp.cpu_count())

    while np.amax(fit_error) > err_bound:
        # get split locations based on number of splits, and split curve & data
        u_i = list(map(tuple, np.repeat(np.linspace(0, 1, splits + 1), 2)[1:-1].reshape(
            (-1, 2))))  # get initial split locations
        try:
            results = pool.amap(get_curve_section, [curve] * splits, [filtered_profile_pts] * splits, u_i)
        except GeomdlException:
            raise GeomdlException("Cannot split the curve at these locations. Check the number of splits.")
        while not results.ready():
            sleep(1)
        
        curves_split, profiles_split, uk_split = zip(*results.get())

        # ******************** KNOT INSERTION ROUTINE ********************
        results1 = pool.amap(adding_knots, profiles_split, curves_split, [err_bound] * splits, uk_split)
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
        if np.amax(rcurve_err) < err_bound:
            curve = rcurve
            fit_error = rcurve_err
            print('new curve')
        else:
            print('kept curve, new loop')
            cp_size_start += 10
            curve = Mfitting.approximate_curve(fit_pts, degree, ctrlpts_size=cp_size_start)
            fit_error = get_error(filtered_profile_pts, curve)
            if cp_size_start > max_cp_size:
                warn("\nCould not fit to given error bound. Try increasing max_error. \nReturning curve...")
                break
        # for debugging
        # pause = input("End of loop. Press any key to continue:\n")

    fit_time = dt.now() - pbs_start
    print(f"Final fit max error: {np.amax(fit_error)} nm\nFitting runtime: {fit_time}\n")
    curve_plotting(profile_pts, curve, err_bound, med_filter=med_filter, sep=False, title='Final Fit', save_to=save_to, plot_show=True)
    return curve_0, curve, [init_time, fit_time]


def get_profiles(filename):
    data_xyz = pd.read_csv(filename, delimiter=',', names=['x', 'y', 'z'])
    num_profile = len(set(data_xyz['y'].values))
    profile_arrs = np.array_split(data_xyz, num_profile)

    return profile_arrs


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-f', '--filepath', help='filepath of input file or directory', type=str)
    parser.add_argument('-s', '--save-dir', type=str, help='saves figures if not empty', default="output_files")
    parser.add_argument('-e', '--error-bound', type=float, help='fitting bound, in nanometers', default=1.0)
    parser.add_argument('-cp', '--ctpts-size', type=int, help='initial control points size for fitting', default=30)
    parser.add_argument('-fw', '--filter-window', type=int, help='filter window size for noisy data', default=0)

    return parser.parse_args()


def fit_single(fpath, save_to, err_bound, cp_size_start, med_filter):
    """ Fits to a single profile in file

    """

    if save_to != '' and not os.path.isdir(save_to):
        os.makedirs(save_to)

    profile_df = get_profiles(fpath)[0]
    # see if it's saved in meters, convert to nm for consistency
    if profile_df['z'].max() <= 10 ** -6:
        profile_df *= 10 ** 9
        if profile_df['z'].min() < 0:
            profile_df['z'] -= profile_df['z'].min()

    # single profile only
    curve_0, curve, times = pbs_iter_curvefit(profile_df, cp_size_start=cp_size_start, err_bound=err_bound, med_filter=med_filter, save_to=save_to)

    if save_to != '':
        with open(f"{os.path.join(save_to, os.path.splitext(os.path.basename(fpath))[0])}.log", 'w') as logfile:
            logfile.write(f'Filename: {os.path.normpath(fpath)}\n'
                          f'Number of points: {len(profile_df)}\n'
                          f'Initial control points: {cp_size_start}\n'
                          f'Initial fit time: {times[0]}\n'
                          f'Filter window: {med_filter}\n'
                          f'Final control points: {curve.ctrlpts_size}\n'
                          f'Total fitting time: {times[1]}')

    return curve_0, curve


if __name__ == '__main__':
    start = dt.now()

    params = parse_args()
    # save_dir = params.save_dir
    # max_error = params.error_bound
    # ctpts_size = params.ctpts_size
    # filter_window = params.filter_window

    params.filepath = "data_new/experimental/experimental_test_xyz.csv"
    save_dir = "output_files/experimental_new"
    max_error = 1.0
    ctpts_size = 80
    filter_window = 46

    if os.path.isdir(params.filepath):
        source = params.filepath
        supported = ['.xyz', '.csv', '.txt']
        files = [fi for fi in sorted(os.listdir(params.filepath)) if os.path.splitext(fi)[1] in supported]

        fitted_curves = []

        for fi in files:
            print(f'Fitting to file {fi}...')
            file = os.path.join(source, fi)
            fitted_curves.append(fit_single(file, save_dir, max_error, ctpts_size, filter_window))

    elif os.path.isfile(params.filepath):
        filter_window = params.filter_window
        initial_curve, fitted_curve = fit_single(params.filepath, save_dir, max_error, ctpts_size, filter_window)

    else:
        raise FileNotFoundError('The input path does not exist!')

    print(f'Total runtime: {dt.now() - start}\n')

    # ## for looping through dir
    # source = "data_new/time_trials"
    # files = [file for file in sorted(os.listdir(source)) if '.csv' in file]
    # for file in files:
    #     print(f'Fitting to {os.path.basename(file)}...')
    #
    #     profile_df = get_profiles(os.path.join(source, file))[0]
    #
    #     # see if it's saved in meters, convert to nm for consistency
    #     if profile_df['z'].max() <= 10**-6:
    #         profile_df *= 10 ** 9
    #         if profile_df['z'].min() < 0:
    #             profile_df['z'] -= profile_df['z'].min()
    #
    #     if 'noisy' in file or 'experimental' in file:
    #         filter_window = 30
    #     else:
    #         filter_window = 0
    #
    #     start_time = dt.now()
    #     # single profile only
    #     curve = pbs_iter_curvefit(profile_df, cp_size_start=ctpts_size, max_error=max_err, filter_size=filter_window, save_to=save_dir)
    #     runtime = dt.now() - start_time
    #
    #     with open(f"{os.path.join(save_dir, os.path.splitext(os.path.basename(file))[0])}.log", 'w') as logfile:
    #         logfile.write(f'Filename: {file}\n'
    #                       f'Number of points: {len(profile_df)}\n'
    #                       f'Initial control points: {ctpts_size}\n'
    #                       f'Filter window: {filter_window}\n'
    #                       f'Final control points: {curve.ctrlpts_size}\n'
    #                       f'Runtime: {runtime}')
    #
    #     print(f'Runtime: {runtime}\nDone with file {os.path.splitext(file)[0]}! \n')

    # putting the curves together
    # curves_u = []
    # for profile_df in arr_splitting:
    #     # profile_df *= (10 ** 9)   # if data is in meters
    #     c = pbs_iter_curvefit(profile_df, cp_size_start=cpts_size, max_error=max_err, filter_size=filter_window)
    #     curve_plotting(profile_df, c, max_err, med_filter=filter_window, title="Final BSpline Fit", save_to=save_dir)
    #     curves_u.append(c)
    # if len(curves_u) > 1:
    #     surf_plot(curves_u)

    runtime = dt.now() - start_time
    print("Runtime: ", runtime)
