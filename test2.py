import cv2
import numpy as np
import matplotlib.pyplot as plt

for i in range(1, 33):
    img = cv2.imread(f'./experiment/{i}.jpg', 0)
    dilated_img = cv2.dilate(img, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = diff_img.copy() # Needed for 3.x compatibility
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap="gray")
    plt.title("gray"), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(bg_img, cmap="gray")
    plt.title("bg_img"), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(diff_img, cmap="gray")
    plt.title("diff"), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(norm_img, cmap="gray")
    plt.title("norm"), plt.xticks([]), plt.yticks([])
    plt.show()