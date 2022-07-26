"""
    Fits input data set to NURBS curves within error threshold and constructs a NURBS surface from the refined curves.
"""

# imports
from geomdl import construct, operations, helpers
# from geomdl.visualization import VisMPL as vis
import matplotlib.pyplot as plt
import numpy as np
from geomdl_mod import Mfitting, Mconstruct
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
import shelve
from scipy import signal as sg


def main():
    filename = 'test_run1_end.out'  ## Set this before running to not overwrite! ##
    # # load data_v
    data_v = np.loadtxt("ground_truth_08_14_20_1kpts3liens.csv", delimiter=',')
    data_v[:, [0, 1]] = data_v[:, [1, 0]] * 100  # assume z-unit = (x,y)-unitE-3 (scaling just for convenience)
    #   Exporting to XYZ data_v from gwyddion gives you a left-handed coordinate system so I swap the x and y cols such that
    #   (x, y) => (u, v)

    # dimensions of data_v and inputs
    profiles = 3  # *** is there a good way to extract this from the data_v array? ***
    tot_points = len(data_v)
    y_dim = int(tot_points / profiles)
    size_u = profiles
    size_v = y_dim
    deg_u = 3
    deg_v = 3
    ctrlpts_size_u = 30
    ctrlpts_size_v = 30
    err_bound = 10
    width_add_knots = 0.04  # for knot refinement/insertion
    num_add_knots = 10  # number of knots to add on EACH SIDE of the high error

    # split the data_v into a dictionary of scan scans for curve fitting
    data_profiles = np.split(data_v, profiles)  # splits data_v array into list of arrays. "profiles" sets the # of splits
    profile_dict = {}
    key = 0
    for scan in data_profiles:  # map to a tuple of tuples so it can be used in Mfitting
        profile_dict[key] = tuple(map(tuple, scan))  # maps each array in the list to a tuple of tuples
        key += 1

    # Fit NURBS curves_v to each scan line *** the profile scans are along the v-direction if (x, y) => (u, v) ***
    curves_v = []
    for key in profile_dict:  # using profile_dict because it's the tuples
        fit_pts = profile_dict[key]
        curve = Mfitting.approximate_curve(fit_pts, deg_v, ctrlpts_size=ctrlpts_size_v)
        curves_v.append(curve)

    # Construct NURBS surface from curves_v *** the profile scans are along the v-direction if (x, y) => (u, v) ***
    # surf_v = Mconstruct.construct_surface('v', curves_v, degree=3)

    # Grab some middle curves_v' indices for plotting, and a sample of data_v points
    one_third = int(len(curves_v) / 3)
    two_thirds = int(len(curves_v) * (2 / 3))
    idx = np.random.randint(0, len(data_v), int(len(data_v) / 5))
    idx = np.unique(idx)

    # Plot the curves_v + surface alongside data_v points
    cv0pts = np.array(curves_v[0].evalpts)
    cv1_3pts = np.array(curves_v[one_third].evalpts)
    cv2_3pts = np.array(curves_v[two_thirds].evalpts)
    cv_1pts = np.array(curves_v[-1].evalpts)
    # surfpts = np.array(surf_v.evalpts)
    datapts = data_v[idx, :]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(datapts[:, 0], datapts[:, 1], datapts[:, 2], color='red')
    ax.plot(cv0pts[:, 0], cv0pts[:, 1], cv0pts[:, 2], color='blue', linewidth=3.0)
    ax.plot(cv1_3pts[:, 0], cv1_3pts[:, 1], cv1_3pts[:, 2], color='blue', linewidth=3.0)
    ax.plot(cv2_3pts[:, 0], cv2_3pts[:, 1], cv2_3pts[:, 2], color='blue', linewidth=3.0)
    ax.plot(cv_1pts[:, 0], cv_1pts[:, 1], cv_1pts[:, 2], color='blue', linewidth=3.0)
    # ax.plot_trisurf(surfpts[:, 0], surfpts[:, 1], surfpts[:, 2], color='orange', alpha=0.75, linewidth=4)
    plt.show()

    # Calculate the residuals between the curves and data points and if it's above err_bound, insert new unique knots
    #   into the knot vector and re-fit to the data.
    rcurves = []
    max_err_list = []
    max_density = 3
    new_err_list = []
    index = 0
    while index < len(curves_v):
        evl_points = data_profiles[index]  # using data_profiles because it's an ndarray and I can call only y values easier
        evl_points_list = list(evl_points[:, 1])  # it's a curve along v so since y => v, use y values in evaluate_list
        evl_points_norm = []
        for point in evl_points_list:  # normalize the points values to between 0 and 1
            norm = (float(point) - min(evl_points_list)) / (max(evl_points_list) - min(evl_points_list))
            evl_points_norm.append(norm)
        curve = curves_v[index]
        crv_pts = np.array(curve.evaluate_list(evl_points_norm))
        residuals = np.absolute(evl_points[:, 2] - crv_pts[:, 2])
        kv_og = curve.knotvector
        # residuals = []
        # row = 0
        # while row < len(evl_points):
        #     error = sqrt((evl_points[row, 0] - crv_pts[row, 0]) ** 2 + (evl_points[row, 1] - crv_pts[row, 1]) ** 2 + (evl_points[row, 2] - crv_pts[row, 2]) ** 2)
        #     residuals.append(error)
        #     row += 1
        max_density = 0
        density = 1
        if max_density >= 1:  # old code that changes refinement (does not work in reasonable time for >1000 pts?
            max_err = np.amax(residuals)
            max_err_list.append(max_err)
            rcurve = None
            if max_err > err_bound:
                while max_err > err_bound and density < max_density:
                    ctpts = curve.ctrlptsw if curve.rational else curve.ctrlpts
                    rctpts, rkv = helpers.knot_refinement(curve.degree, curve.knotvector, ctpts, density=density)
                    fit_pts = profile_dict[index]
                    rcurve = Mfitting.approximate_curve(fit_pts, curve.degree, kv=rkv, ctrlpts_size=len(rctpts))
                    rcrv_pts = np.array(rcurve.evaluate_list(evl_points_norm))
                    residuals = evl_points[:, 2] - rcrv_pts[:, 2]
                    # residuals = []
                    # row = 0
                    # while row < len(evl_points):
                    #  was throwing error here
                    #  error = sqrt(
                    #     (evl_points[row, 0] - rcrv_pts[row]) ** 2 + (evl_points[row, 1] - rcrv_pts[row]) ** 2 + (
                    #             evl_points[row, 2] - rcrv_pts[row, 2]) ** 2)
                    # residuals.append(error)
                    # row += 1
                    max_err = np.amax(residuals)
                    density += 1
            if rcurve is None:
                rcurves.append(curve)
            else:
                rcurves.append(rcurve)
            new_err_list.append(max_err) \
                # selective refinement (liam)
        else:
            err_peaks, err_peaks_info = sg.find_peaks(residuals,
                                                      height=err_bound)  # find error peaks greater than 10 nm (steps)
            err = np.max(err_peaks_info['peak_heights'])
            kv_new = kv_og  # holder for new knots
            ctrl_pts_old = curve.ctrlptsw if curve.rational else curve.ctrlpts
            loop_count = 0
            while err > err_bound and loop_count < 3:
                ind = 0
                for pk in err_peaks:
                    # knts_to_add = np.linspace(pk/len(crv_pts)*0.995, pk/len(crv_pts)*1.005, 10)  # add 20 knots in
                    #                                                                            # normalized v coord +/-5%
                    #                                                                            # from the index of the peak. (Assuming we eval each data pt (fine for now))
                    knts_to_add = np.random.normal(pk / len(evl_points), 0.02 * pk / len(evl_points), (20, 1))
                    for knt in knts_to_add:
                        if knt not in kv_new:  # knotty puns
                            kv_new = helpers.knot_insertion_kv(kv_new, knt[0], 1, 1)  # add new knots to the kv from OG or last iter
                            if ind == 0:
                                ctrl_pts_new = helpers.knot_insertion(3, kv_new, ctrl_pts_old, knt[0])
                            ctrl_pts_new = helpers.knot_insertion(3, kv_new, ctrl_pts_new, knt[0])  # update ctrl pts vector so old ctrl pts stay with their corresponding knts (does this matter?)
                            ind += 1

                fit_pts = profile_dict[index]
                kv_new = kv_new.sort()
                rcurve = Mfitting.approximate_curve(fit_pts, 3, kv=kv_new, ctrlpts_size=len(ctrl_pts_new))
                rcrv_pts = np.array(rcurve.evaluate_list(evl_points_norm))
                rcrv_residuals = np.absolute(evl_points[:, 2] - rcrv_pts[:, 2])
                err_peaks, err_peaks_info = sg.find_peaks(rcrv_residuals,
                                                          height=err_bound)  # find error peaks greater than 10 nm (steps)
                err = np.max(err_peaks_info['peak_heights'])
                kv_new = rcurve.knotvector  # holder for new knots
                ctrl_pts_old = rcurve.ctrlptsw if rcurve.rational else rcurve.ctrlpts
                loop_count += 1

        rcurves.append(rcurve)
        index += 1

    # Create the *refined* NURBS surface
    # rsurf = Mconstruct.construct_surface('v', rcurves, degree=3)

    # Plot the *refined* curves_v + surface alongside data_v points
    rcv0pts = np.array(rcurves[0].evalpts)
    rcv1_3pts = np.array(rcurves[one_third].evalpts)
    rcv2_3pts = np.array(rcurves[two_thirds].evalpts)
    rcv_1pts = np.array(rcurves[-1].evalpts)
    # rsurfpts = np.array(rsurf.evalpts)
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    ax2.scatter(datapts[:, 0], datapts[:, 1], datapts[:, 2], color='red')
    ax2.plot(rcv0pts[:, 0], rcv0pts[:, 1], rcv0pts[:, 2], color='blue', linewidth=3.0)
    ax2.plot(rcv1_3pts[:, 0], rcv1_3pts[:, 1], rcv1_3pts[:, 2], color='blue', linewidth=3.0)
    ax2.plot(rcv2_3pts[:, 0], rcv2_3pts[:, 1], rcv2_3pts[:, 2], color='blue', linewidth=3.0)
    ax2.plot(rcv_1pts[:, 0], rcv_1pts[:, 1], rcv_1pts[:, 2], color='blue', linewidth=3.0)
    # ax2.plot_trisurf(rsurfpts[:, 0], rsurfpts[:, 1], rsurfpts[:, 2], color='orange', alpha=0.75, linewidth=4)

    # plt.show()
    plt.savefig("fit_lgc.png")

    the_shelf = shelve.open(filename)
    for key in dir():
        try:
            the_shelf[key] = globals()[key]
        except TypeError or KeyError:
            print('ERROR shelving: {0}'.format(key))
    the_shelf.close()

    pass


if __name__ == '__main__':
    main()
