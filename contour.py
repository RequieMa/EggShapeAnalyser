import cv2
import numpy as np
import argparse
import random as rng
rng.seed(12345)
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from setting import *

def find_egg_contour(src, threshold):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.blur(src_gray, (3, 3))
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    if DEBUG:
        cv2.imshow("Canny", canny_output)

    contours_full, hierarchy  = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_simple, hierarchy  = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_simple: # loop over one contour area
        for contour_point in contour: # loop over the points
            cv2.circle(src, ((contour_point[0][0], contour_point[0][1])), 2, (0, 255, 0), 1, cv2.LINE_AA)
    if DEBUG:
        cv2.imshow("Contour Points", src)
    return contours_full, contours_simple

parser = argparse.ArgumentParser(description='Code for Creating Bounding boxes and circles for contours tutorial.')
parser.add_argument('--input', help='Path to input image.', default='test2.jpg')
args = parser.parse_args()
src = cv2.imread(cv2.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
# Convert image to gray and blur it

threshold = 100
contours, contours_simple = find_egg_contour(src.copy(), threshold)

for i in range(len(contours)):
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv2.drawContours(src, contours, i, color, thickness=2, lineType=cv2.LINE_AA)

if DEBUG:
    source_window = 'Contours'
    cv2.namedWindow(source_window)
    cv2.imshow(source_window, src)
    cv2.waitKey()
    cv2.destroyAllWindows()

contours_simple = np.array(contours_simple, dtype=np.int32)
_, length, _, coor = contours_simple.shape
contours_simple = contours_simple.reshape((length, coor))
h, w, _ = src.shape

def transform2world(img_points):
    recenter = np.array([
        [w / 2, h / 2],
    ])
    recentered_img = img_points - recenter
    return recentered_img

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

contours_simple = transform2world(contours_simple)
radius, radiances = cart2pol(contours_simple[:, 0], contours_simple[:, 1])
plt.axes(projection = 'polar') 
for r, phi in zip(radius, radiances): 
    plt.polar(phi, r, 'g.') 
plt.show() 