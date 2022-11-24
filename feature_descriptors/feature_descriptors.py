import cv2
import numpy as np

def hog(img, group_size=3, step=2, bin_num=16):
    img = cv2.resize(img, (64,64))
    gx = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
    gy = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy)

    bins = np.int32(bin_num * ang / (2 * np.pi))
    bin_cells = []
    mag_cells = []
    if step is None:
        step = group_size


    mag_cells = [
        mag[i : i + group_size, j : j + group_size]
        for i in range(0, img.shape[0] - group_size, step)
        for j in range(0, img.shape[1] - group_size, step)
    ]
    bin_cells = [
        bins[i : i + group_size, j : j + group_size]
        for i in range(0, img.shape[0] - group_size, step)
        for j in range(0, img.shape[1] - group_size, step)
    ]
    hists = [
        np.bincount(b.ravel(), m.ravel(), bin_num) for b, m in zip(bin_cells, mag_cells)
    ]
    hist = np.hstack(hists)

    return hist