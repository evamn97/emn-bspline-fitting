""" This program takes a point-cloud input of data_v points and fits a B-Spline (NURBS) surface to
    them using global approximation. To improve the fitting at sharp features, we measure the residuals
    and iterate through a method of optimizing the knot vectors to decrease error.
"""

# imports
from geomdl import fitting, exchange, construct, operations, helpers
# from data_generator import create_data
from geomdl.visualization import VisMPL as vis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = np.loadtxt("stepspatt.csv", delimiter=',')
# data_v = pd.read_csv("xyztest.csv", comment='#').to_numpy()
data[:, [0, 1]] = data[:, [1, 0]]
#   Exporting to XYZ data_v from gwyddion gives you a left-handed coordinate system so I swap the x and y cols such that
#   (x, y) => (u, v)

# data_v = pd.read_csv("VTKtestytest.csv", skiprows=6, names=['x', 'y', 'z'], usecols=[0, 1, 2]).to_numpy()
#   By using gwyddion to export the data_v as vtk (Liam's method), the coord frame is: x down, y right/across, z out;
#   This means x-coordinates are scan locations, and y-coordinates are sample locations
#   u-dir corresponds to x, v-dir corresponds to y

profiles = 35  # *** is there a good way to extract this from the data_v array? ***
points = len(data)

y_dim = int(points / profiles)

size_u = profiles
size_v = y_dim
# I'm not sure if the degrees need to be this large but I wrote in my notes that
#   the NURBS book suggests setting p >= r+1 and q >= s+1 where size_u = r and
#   size_v = s.
degree_u = 3
degree_v = 3
# NURBS book says start with a minimum of p+1 and q+1 control points
ctrlpts_size_u = degree_u + 3
ctrlpts_size_v = degree_v + 3

# Global surface approximation
surf = fitting.approximate_surface(data, size_u, size_v, degree_u, degree_v, ctrlpts_size_u=ctrlpts_size_u,
                                   ctrlpts_size_v=ctrlpts_size_v)

# Extract the isoparametric curves_v of the surface for use in deviation check
surf_curves = construct.extract_curves(surf)

plot_extras = [
    dict(
        points=surf_curves['u'][0].evalpts,
        name="u",
        color="cyan",
        size=5
    ),
    dict(
        points=surf_curves['v'][0].evalpts,
        name="v",
        color="magenta",
        size=5
    )
]

# Plot the surface using VisMPL
surf.delta = 0.05
surf.vis = vis.VisSurfWireframe()
surf.render(extras=plot_extras)

# Plot surface points and data_v together using matplotlib
# evalpts = np.array(surf_v.evalpts)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(evalpts[:, 0], evalpts[:, 1], evalpts[:, 2])
# ax.scatter(data_v[:, 0], data_v[:, 1], data_v[:, 2], color="red")
# plt.show()

# Evaluate residuals using surf_v.evaluate_list() and compare to measured points
xy_pts_arr = data[0:y_dim, [0, 1]]
z_pts = data[0:y_dim, 2]
xy_pts_list = list(map(tuple, xy_pts_arr))
pts_eval = np.array(surf.evaluate_list(xy_pts_list))
diff = z_pts - pts_eval[:, 2]
#   **** Need to add cycling through successive points in xy_pts_list ****

# Apply knot refinement (operations applies it over the entire surface)
density_u = 1
density_v = 1
rsurf = operations.refine_knotvector(surf, [density_u, density_v])

# Plot the refined surface with the data_v
# revalpts = np.array(rsurf.evalpts)
# fig2 = plt.figure()
# ax2 = plt.axes(projection='3d')
# ax2.scatter(revalpts[:, 0], revalpts[:, 1], revalpts[:, 2])
# ax2.scatter(data_v[:, 0], data_v[:, 1], data_v[:, 2], color="red")
# plt.show()

# Use helpers.py to selectively apply knot refinement (this function only applies to a single B-spline, not a surface,
#   so how do I use it for the surface??)
#   *** it's called in the code of operations.refine_knotvector(), but I don't understand that part of the code ***
# knot_list = []
# helpers.knot_refinement()

# TODO: THINK ABOUT FITTING B-SPLINES TO EACH PROFILE IN FAST AND SLOW SCAN DIRECTIONS, REFINING EACH SPLINE AS NEEDED WITH HELPERS.KNOT_REFINEMENT AND THEN USE CONSTRUCT.CONSTRUCT_SURFACE TO CREATE THE SURFACE
