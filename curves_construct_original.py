"""
    Fits input data set to NURBS curves within error threshold and constructs a NURBS surface from the refined curves.
"""

# imports
from geomdl import construct, operations, helpers
# from geomdl.visualization import VisMPL as vis
import matplotlib.pyplot as plt
import numpy as np
from geomdl_mod import Mfitting, Mconstruct
from mpl_toolkits.mplot3d import Axes3D

# # load data_v
data_v = np.loadtxt("stepspatt.csv", delimiter=',')
data_v[:, [0, 1]] = data_v[:, [1, 0]] * 100  # assume z-unit = (x,y)-unitE-3 (scaling just for convenience)
#   Exporting to XYZ data_v from gwyddion gives you a left-handed coordinate system so I swap the x and y cols such that
#   (x, y) => (u, v)

# dimensions of data_v and inputs
profiles = 35  # *** is there a good way to extract this from the data_v array? ***
tot_points = len(data_v)
y_dim = int(tot_points / profiles)
size_u = profiles
size_v = y_dim
deg_u = 3
deg_v = 3
ctrlpts_size_u = 30
ctrlpts_size_v = 30
err_bound = 4.8e-06
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

# Calculate the residuals between the curves and data points and if it's above err_bound, insert new unique knots into the knot vector and re-fit to the data.
rcurves = []
max_err_list = []
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
    residuals = evl_points[:, 2] - crv_pts[:, 2]
    max_err = np.amax(residuals)
    max_err_list.append(max_err)
    if max_err > err_bound:
        add_knots = []
        while max_err > err_bound and len(add_knots) < 50:
            err_pt = evl_points_norm[index] # u (or v) location of large error
            add_knots_tmp = []
            k_temp_a = np.linspace(err_pt - width_add_knots, err_pt, num=num_add_knots)
            k_temp_b = k_temp_a + err_pt
            add_knots_tmp.extend(k_temp_a.tolist())
            add_knots_tmp.extend(k_temp_b.tolist())
            add_knots_tmp = list(set(add_knots_tmp))
            for k in add_knots_tmp:
                if k not in curve.knotvector:
                    add_knots.append(k)
            add_knots.sort()
            rcurve_tmp = operations.insert_knot(curve, add_knots, [1])
            fit_pts = profile_dict[index]
            rcurve = Mfitting.approximate_curve(fit_pts, rcurve_tmp.degree, kv=rcurve_tmp.knotvector, ctrlpts_size=rcurve_tmp.ctrlpts_size)
            rcurves.append(rcurve)
            rcrv_pts = np.array(rcurve.evaluate_list(evl_points_list))
            residuals = evl_points[:, 2] - rcrv_pts[:, 2]
            max_err = np.amax(residuals)
        new_err_list.append(max_err)
    index += 1

# Construct NURBS surface from curves_v *** the profile scans are along the v-direction if (x, y) => (u, v) ***
# surf_v = Mconstruct.construct_surface('v', curves_v, degree=3)

# Create the *refined* NURBS surface
# rsurf = Mconstruct.construct_surface('v', rcurves, degree=3)

# Grab some middle curves_v' indices for plotting, and a sample of data_v points
one_third = int(len(curves_v) / 3)
two_thirds = int(len(curves_v) * (2 / 3))
idx = np.random.randint(0, len(data_v), int(len(data_v) / 5))
idx = np.unique(idx)

# Plot the curves_v + surface alongside data_v points
# cv0pts = np.array(curves_v[0].evalpts)
# cv1_3pts = np.array(curves_v[one_third].evalpts)
# cv2_3pts = np.array(curves_v[two_thirds].evalpts)
# cv_1pts = np.array(curves_v[-1].evalpts)
# # surfpts = np.array(surf_v.evalpts)
datapts = data_v[idx, :]
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(datapts[:, 0], datapts[:, 1], datapts[:, 2], color='red')
# ax.plot(cv0pts[:, 0], cv0pts[:, 1], cv0pts[:, 2], color='blue', linewidth=3.0)
# ax.plot(cv1_3pts[:, 0], cv1_3pts[:, 1], cv1_3pts[:, 2], color='blue', linewidth=3.0)
# ax.plot(cv2_3pts[:, 0], cv2_3pts[:, 1], cv2_3pts[:, 2], color='blue', linewidth=3.0)
# ax.plot(cv_1pts[:, 0], cv_1pts[:, 1], cv_1pts[:, 2], color='blue', linewidth=3.0)
# ax.plot_trisurf(surfpts[:, 0], surfpts[:, 1], surfpts[:, 2], color='orange', alpha=0.75, linewidth=4)

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
plt.savefig("ccog_refined_fit.png")

# TODO: fit the two scan directions to surf_u and surf_v, then take the fitted control points in u for surf_u,
#  and in v for surf_v, and apply them to an overall surf_v
#  TODO: take the fitted curves_v in both directions and measure difference between them at crossing points
