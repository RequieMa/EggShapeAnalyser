import matplotlib.pyplot as plt 
import numpy as np
from matplotlib import collections  as mc
from scipy import fft

def sort_method(points, reference_point):
	points = np.array(points)
	distance = np.linalg.norm(points - reference_point)
	# print(points, distance)
	return distance

def get_sorted_points(pts):
    midpoints = pts.copy()
    sorted_points = [midpoints[0]]
    midpoints.pop(0)
    for i in range(len(midpoints) - 1):
        reference_point = sorted_points[i]
        rest_points = midpoints
        rest_points.sort(key=lambda x: sort_method(x, np.array(reference_point)))
        cloest_point = rest_points[0]
        sorted_points.append(cloest_point)
        idx = midpoints.index(cloest_point)
        midpoints.pop(idx)
    return sorted_points

if __name__ == "__main__":
    midpoints = [[305.0, 244.5], [293.5, 237.0], [274.5, 367.5], [270.5, 373.5], [229.5, 391.0], [216.0, 396.0], [302.0, 269.0], [295.0, 271.0]]
    reference_point = midpoints[0]
    reference_array = np.array(reference_point)
    rest_points = midpoints[1:]
    rest_points.sort(key=lambda x: sort_method(x, reference_array))
    sorted_points = [reference_point] + rest_points
    sorted_points = np.array(sorted_points)

    x_list = sorted_points[:, 0]
    y_list = sorted_points[:, 1]

    complex_mdpts = [[x + 1j * y] for x, y in zip(x_list, y_list)]
    coefs = fft.fft(complex_mdpts, axis=0)
    N = len(coefs)
    print(f"coeffs {N}:\n{coefs[:5]}")

    # function in terms of t to trace out curve
    # m = N
    # lim = int(m / 2)
    # def S(k):
    #     ftx, fty = 0, 0
    #     # for i in range(-lim, lim + 1):
    #     for idx, n in enumerate(np.linspace(-lim, lim, N)):
    #         func = (coefs[idx] * np.exp(1j * 2*np.pi * k / N * n))
    #         ftx += func.real[0]
    #         fty += func.imag[0]
    #     return [ftx / n, fty / n]
        

    # lines = [] # store computed lines segments to approximate function
    # pft = S(0) # compute first point
    # for k in np.linspace(0, N, N): 
    #     cft = S(k)
    #     lines.append([cft, pft])
    #     pft = cft

    # lc = mc.LineCollection(lines)
    # fig, ax = plt.subplots()
    # ax.scatter(x_list, y_list, s=50, marker="x", color='y') 
    # ax.add_collection(lc)
    # ax.autoscale()
    # ax.margins(0.1)
    # plt.show()