import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import multiprocessing as mp
from geomdl_mod import Mfitting

filename = "lines_patt.csv"
data_2d = np.loadtxt(filename, delimiter=',')
profiles = len(set(data_2d[:, 1]))
data_3d = np.reshape(data_2d, (profiles, -1, 3))
deg_v = 3
ctrlpts_size_v = 30


def gen_curve_v(single_curve):
    fit_pts_arr = np.squeeze(single_curve)
    fit_pts = list(map(tuple, fit_pts_arr))
    curve = Mfitting.approximate_curve(fit_pts, deg_v, ctrlpts_size=ctrlpts_size_v)
    return curve


if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as pool:
        curves_v = pool.map(gen_curve_v, [s_curve for s_curve in data_3d])
    print("done parallelizing")
    cv0pts = np.array(curves_v[0].evalpts)
    cv1_3pts = np.array(curves_v[3].evalpts)
    cv2_3pts = np.array(curves_v[6].evalpts)
    cv_1pts = np.array(curves_v[-1].evalpts)

    fig = plt.figure(figsize=(12.8, 9.6))
    ax = plt.axes(projection='3d')
    ax.plot(data_2d[:, 0], data_2d[:, 1], data_2d[:, 2], color='red')
    ax.plot(cv0pts[:, 0], cv0pts[:, 1], cv0pts[:, 2], color='blue', linewidth=3.0)
    ax.plot(cv1_3pts[:, 0], cv1_3pts[:, 1], cv1_3pts[:, 2], color='blue', linewidth=3.0)
    ax.plot(cv2_3pts[:, 0], cv2_3pts[:, 1], cv2_3pts[:, 2], color='blue', linewidth=3.0)
    ax.plot(cv_1pts[:, 0], cv_1pts[:, 1], cv_1pts[:, 2], color='blue', linewidth=3.0)
    plt.show()