from datetime import datetime as dt, timedelta
from time import sleep

from utils import *


# def get_error_z(profile_pts, curve_obj, uk=None):
#     """ Gets the Z fitting error for a single curve.
#         :param profile_pts: dataframe of profile points
#         :param curve_obj: fitted curve object
#         :param uk: parametric coordinates
#         :return: fitted curve Z error, ndarray
#         """
#     import numpy as np  # pathos raises an 'np undefined' error without this
#     points = profile_pts.values  # get array of points from df
#     if uk is not None:
#         eval_idx = uk
#     else:
#         eval_idx = Mfitting.compute_params_curve(list(map(tuple, points)))
#         if list(np.asarray(curve_obj.knotvector)[[0, -1]]) != [0.0, 1.0]:
#             eval_idx = normalize(eval_idx, low=curve_obj.knotvector[0], high=curve_obj.knotvector[-1])
#         eval_idx = list(eval_idx)
#
#     # ZeroDivisionError sometimes comes up here if there's an extra unmatching knot at the end of the knot vector
#     # TODO: idk where that extra knot comes from
#     curve_points = np.array(curve_obj.evaluate_list(eval_idx))
#     z_error = np.sqrt(np.square(np.subtract(points[:, 2], curve_points[:, 2])))
#
#     return z_error
#
#
# def plot_curve_only(crv, title="B-spline curve only"):
#     """
#     Plots the curve only on the x-axis
#
#     :param crv: curve to plot
#     :param title: title of the plot
#     :return: None
#     """
#     crv.delta = 0.001
#     crv.evaluate()
#     crv_pts = np.array(crv.evalpts)
#     ct_pts = np.array(crv.ctrlpts)
#
#     fig, ax = plt.subplots(figsize=(40, 15))
#     ax.grid(True)
#     ax.plot(crv_pts[:, 0], crv_pts[:, 2], label='Fitted Curve', c='red', linewidth=2)
#     ax.plot(ct_pts[:, 0], ct_pts[:, 2], label='Control Points', marker='+', c='orange', linestyle='--', linewidth=2, alpha=0.7)
#
#     scaled_kv = normalize(crv.knotvector, low=np.amin(crv_pts[:, 0]), high=np.amax(crv_pts[:, 0]))
#     bottom = np.amin(crv_pts[:, -1]) - (np.amax(crv_pts[:, -1]) - np.amin(crv_pts[:, -1])) * 0.2
#     ax.hist(scaled_kv, bins=(int(len(crv_pts[:, 0]))), bottom=bottom, label='Knot Locations')
#     ax.set(xlabel='Lateral Position X [nm]', ylabel='Height Z [nm]',
#            title=(title.upper() + ': Control Points={}'.format(crv.ctrlpts_size)))
#     ax.legend(loc="upper right")
#     fig.tight_layout()
#     plt.show()
#
#
# def get_section_multi(profile_pts: pd.DataFrame, splits: int, cp_size: int):
#     """
#         Returns a list of data and u_k sections defined by low and high u_k values. End points are overlapping
#         :param profile_pts: data points the input curve is fitted to
#         :param splits: how many sections to return
#         :param cp_size: number of total control points (must be divisible by splits)
#         :return: list of curve section (with clamped exterior knots), list of section data points
#         """
#     # cp_size_section = int(cp_size / splits)
#     degree = len(profile_pts.columns)
#     cp_size = cp_size - (degree * (splits - 1))   # splitting the curve & re-merging adds p knots at each split point, so we need to reduce cp_size
#
#     profiles_split = np.array_split(profile_pts, splits)
#     p_join = [p.values[-1] for p in profiles_split][:-1]
#
#     u_k = Mfitting.compute_params_curve(list(map(tuple, profile_pts.values)))
#     uk_split = np.array_split(u_k, splits)
#     u_join = [u[-1] for u in uk_split][:-1]
#
#     kv = Mfitting.compute_knot_vector2(degree, len(profile_pts), cp_size, u_k)
#     i_ksplit = [find_nearest(u, kv, mode='greater than')[0] for u in u_join]
#     kv_split = np.array_split(kv, i_ksplit)
#
#     for s in range(splits - 1):
#         p0 = p_join[s]
#         u0 = u_join[s]
#         uk_split[s + 1] = np.append(uk_split[s + 1], u0)
#         df_temp = pd.DataFrame(columns=['x', 'y', 'z'])
#         loc = min(profiles_split[s + 1].index) - 1
#         df_temp.loc[loc] = p0.tolist()
#         profiles_split[s + 1] = pd.concat((df_temp, profiles_split[s + 1]))
#         kv_split[s] = np.concatenate((kv_split[s], np.repeat(u0, degree + 1)))
#         kv_split[s + 1] = np.concatenate((np.repeat(u0, degree + 1), kv_split[s + 1]))
#
#     return profiles_split, uk_split, kv_split
#
#
# def section_fit(profile_pts, kv):
#     """
#     Fits a section of data points using a given  u_k parameterization, and knot vector
#     :param profile_pts: profile data points
#     :type profile_pts: pandas.DataFrame
#     :param kv: knot vector to use for fitting
#     :return: refined curve
#     """
#     from geomdl_mod import Mfitting  # pathos raises an 'undefined' error without this
#
#     degree = len(profile_pts.columns)
#     kv.sort()
#     temp_kv = list(normalize(kv))
#
#     # TODO: how to do error handling for this? (ValueError when fitting fails)
#     temp_curve = Mfitting.approximate_curve(list(map(tuple, profile_pts.values)),
#                                             degree, kv=temp_kv)
#
#     curve_section = BSpline.Curve(normalize_kv=False)
#     curve_section.degree = degree
#     curve_section.ctrlpts = temp_curve.ctrlpts
#     curve_section.knotvector = list(kv)
#     curve_section.delta = 0.001
#     curve_section.evaluate()
#     # curve_plotting(profile_pts, curve_section, 2.0)
#
#     return curve_section
#
#
# def merge_curves_multi2(args):
#     """
#     Merges two or more curves into a single curve
#
#     :param args: curves to merge
#     :return: merged curve
#     """
#     if len(args) < 2:
#         raise ValueError("At least two curves must be specified in args")
#
#     p = args[0].degree
#     merged_curve = BSpline.Curve(normalize_kv=False)
#     ctrlpts_new = []
#     join_knots = []
#     kv_new = args[0].knotvector[:p + 1]
#     s = []
#
#     for c in args:
#         ctrlpts_new = ctrlpts_new + c.ctrlpts
#         kv_new = sorted(kv_new + c.knotvector[p + 1:])  # ensures rule m = n + p + 1
#         join_knots.append(c.knotvector[-1])
#     join_knots = join_knots[:-1]
#     for j in join_knots:
#         s_i = helpers.find_multiplicity(j, kv_new)
#         s.append(s_i)
#
#     kv_new = list(np.asarray(kv_new).astype(float))
#
#     num_knots = len(ctrlpts_new) + p + 1  # assuming all cpts are kept, find the required number of knots from m = n + p + 1
#     if num_knots != len(kv_new):
#         raise ValueError("Something went wrong with getting the merged knot vector. Check knot removals.")
#
#     merged_curve.degree = p
#     merged_curve.ctrlpts = ctrlpts_new
#     merged_curve.knotvector = kv_new
#     merged_curve.delta = 0.001
#     merged_curve.evaluate()
#
#     where_s = np.where(np.asarray(s) > p)[0]  # returns a 1-dim tuple for some reason
#     for i in where_s:
#         # for large datasets, the knot deletions that occur during merging will blow up the fit error
#         # we leave the multiplicity at k = p, which is the maximum allowed value
#         delete = s[i] - p
#         if delete > 0:
#             merged_curve = Mop.remove_knot(merged_curve, [join_knots[i]], [delete])
#
#     return merged_curve


# def pbs_iter_curvefit3(profile_pts, degree=3, cp_size_start=80, max_error=1.0):
#     """
#     Iterative curve fit for a single profile, adding knots each time based on max error limit
#     Uses parallel splitting based on the number of computer cpus.
#     Includes median filter for noisy data.
#
#     :param profile_pts: pandas dataframe of profile data points
#     :param degree: degree of fitted curve, default=3
#     :param cp_size_start: number of control points to start with
#     :param max_error: maximum error bound, in nanometers
#     :return: fitted curve object
#     """
#     pbs_start = dt.now()
#     fit_pts = list(map(tuple, profile_pts.values))
#
#     max_cp_size = max((len(fit_pts) - degree - 11), int((len(fit_pts) / 10)))  # ensures greatest # knots is len(fit_pts) - 10
#     assert (cp_size_start < max_cp_size), "cp_size_start must be smaller than maximum number of control points. \nGot cp_size_start={}, max_cp_size={}.".format(cp_size_start, max_cp_size)
#     # too many splits will mess up the fit because of the knot deletion
#     # that occurs during curve section merging
#     splits = mp.cpu_count() if mp.cpu_count() < 13 else 13
#     pool = Pool(mp.cpu_count())
#     profiles_split, uk_split, kv_split = get_section_multi(profile_pts, splits, cp_size_start)
#     results = pool.amap(section_fit, profiles_split, kv_split)
#     while not results.ready():
#         sleep(1)
#     curves_split = results.get()
#     curve = merge_curves_multi2(curves_split)
#     fit_error = get_error(profile_pts, curve)
#
#     print(f"{dt.now() - pbs_start} initial fit\ninitial max error = {np.amax(fit_error)}")
#
#     curve_plotting(profile_pts, curve, max_error, sep=True, title='Initial Curve Fit')  # for debugging
#     add_knots = 1
#     unchanged_loops = 0
#     final = False
#     secondary = False  # secondary knot selection method
#
#     loop_times = []
#
#     while final is False:
#
#         loop_start = dt.now()
#
#         if np.amax(fit_error) < max_error:
#             final = True
#             add_knots = 0
#             splits = np.ceil(mp.cpu_count() / 3).astype(int) if np.ceil(mp.cpu_count() / 3) >= 3 else 3
#             u_i = list(map(tuple, np.repeat(np.linspace(0, 1, splits + 1), 2)[1:-1].reshape((-1, 2))))
#             try:
#                 results = pool.amap(get_curve_section, [curve] * splits, [profile_pts] * splits, u_i)
#                 temp = results.get()
#             except GeomdlException:
#                 raise GeomdlException("Cannot split the curve at these locations. Check the number of splits.")
#             while not results.ready():
#                 sleep(1)
#
#             curves_split = [c for (c, _, _) in temp]
#             profiles_split = [p for (_, p, _) in temp]
#             uk_split = [u for (_, _, u) in temp]
#
#         results1 = pool.amap(adding_knots, profiles_split, curves_split,
#                              [add_knots] * splits, [max_error] * splits,
#                              uk_split, [secondary] * splits)
#         while not results1.ready():
#             sleep(1)
#         rcurves_list = results1.get()
#
#         section_err = []
#         for i in range(len(rcurves_list)):
#             rc = rcurves_list[i]
#             pr = profiles_split[i]
#             err = np.amax(get_error(pr, rc))
#             section_err.append(err)
#
#         rcurve = merge_curves_multi2(rcurves_list)
#         rknots_set = sorted(list(set(rcurve.knotvector)))
#         results2 = pool.amap(helpers.find_multiplicity, rknots_set, [rcurve.knotvector] * len(rcurve.knotvector))
#         while not results2.ready():
#             sleep(1)
#         s = results2.get()
#         delete = list(np.where(np.asarray(s) > degree)[0])[1:-1] + list(np.where(np.asarray(s) > degree + 1)[0])
#         for d in delete:
#             k = rknots_set[d]
#             if k == 0.0 or k == 1.0:
#                 if s[d] > (degree + 1):
#                     rcurve = Mop.remove_knot(rcurve, [k], [s[d] - degree - 1])
#             else:
#                 rcurve = Mop.remove_knot(rcurve, [k], [s[d] - degree])
#
#         rcurve_err = get_error(profile_pts, rcurve)
#
#         # print("add knots = {}\nsecondary = {}".format(add_knots, secondary))
#         # print("Max error for sections = {}\nMax error for rf_curve = {}\nMax error for og_curve = {}\n".format(np.amax(section_err),
#         #                                                                                                        np.amax(rcurve_err),
#         #                                                                                                        np.amax(fit_error)))
#
#         if np.amax(rcurve_err) < np.amax(fit_error):
#             # if np.amax(rcurve_err) > max_error:
#             #     curve_plotting(profile_pts, rcurve, max_error, title="New Curve Plot")
#             print(f"refined max error = {np.amax(rcurve_err)}")
#             curve = rcurve
#             curves_split = rcurves_list
#             fit_error = rcurve_err
#             unchanged_loops = 0
#             add_knots = 1
#             secondary = False
#         else:
#             # curve_plotting(profile_pts, rcurve, max_error, title="Rejected Curve Plot")
#             unchanged_loops += 1
#             add_knots += 1
#
#         if unchanged_loops > 2:
#             # sometimes the alg gets stuck with the regular knot generation so
#             # this will reset it to the secondary knot selection method, which is randomized
#             secondary = True
#
#         loop_times.append(dt.now() - loop_start)
#     print(f"average loop time: {sum(loop_times, timedelta(0)) / len(loop_times)}\nfinal max error: {np.amax(fit_error)}\ncurve fit time: {dt.now() - pbs_start}\n")
#
#     return curve


if __name__ == '__main__':
    start_time = dt.now()
    #
    # filename = "data/profile_test_xyz3.csv"
    # data_2d = pd.read_csv(filename, delimiter=',', names=['x', 'y', 'z'])
    # profiles = len(set(data_2d['y'].values))
    # arr_splitting = np.array_split(data_2d, profiles)
    #
    # deg = 3
    # cpts_size = 120
    # max_err = 1.0  # nanometers
    # filter_window = 46
    # save_dir = ""  # os.path.join("data", "experimental-results")
    #
    # profile_df = arr_splitting[0] * (10 ** 9)
    # plot_data_only(profile_df, filter_window, save_to=save_dir)
    #
    # curves_u = []
    # for profile_df in arr_splitting:
    #     profile_df *= (10 ** 9)   # if data is in meters
    #     c = pbs_iter_curvefit3(profile_df, cp_size_start=cpts_size, max_error=max_err)
    #     curve_plotting(profile_df, c, max_err, med_filter=filter_window, title="Final BSpline Fit", save_to=save_dir)
    #     curves_u.append(c)
    #
    # if len(curves_u) > 1:
    #     surf_plot(curves_u)
    #
    end_time = dt.now()
    runtime = end_time - start_time
    print("Program Runtime: ", runtime)
