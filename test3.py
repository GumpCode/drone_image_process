import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("./experiment/5.jpg", 0)
h, w = np.shape(img)
img = cv2.resize(img, (h//2, w//2))
h, w = np.shape(img)
back = img.copy()
half_w = 2
win_size = 2 * half_w + 1


def compute_b(window, n):
    h_, w_ = np.shape(window)
    count = 0
    sum = 0
    for r in range(h_):
        for c in range(w_):
            if window[r, c] <= 250:
                sum += window[r, c]
                count += 1
    return sum // count


for row in range(half_w, h - half_w - 1):
    for col in range(half_w, w - half_w - 1):
        back[row, col] = compute_b(img[row-half_w:row+half_w, col-half_w:col+half_w], win_size)

plt.imshow(back, "gray")
plt.show()