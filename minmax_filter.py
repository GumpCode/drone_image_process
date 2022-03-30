import cv2
import numpy as np
import matplotlib.pyplot as plt


def max_filtering(N, I_temp):
    wall = np.full((I_temp.shape[0]+(N//2)*2, I_temp.shape[1]+(N//2)*2), -1)
    wall[(N//2):wall.shape[0]-(N//2), (N//2):wall.shape[1]-(N//2)] = I_temp.copy()
    temp = np.full((I_temp.shape[0]+(N//2)*2, I_temp.shape[1]+(N//2)*2), -1)
    for y in range(0,wall.shape[0]):
        for x in range(0,wall.shape[1]):
            if wall[y,x]!=-1:
                window = wall[y-(N//2):y+(N//2)+1,x-(N//2):x+(N//2)+1]
                num = np.amax(window)
                temp[y,x] = num
    A = temp[(N//2):wall.shape[0]-(N//2), (N//2):wall.shape[1]-(N//2)].copy()
    return A


def min_filtering(N, A):
    wall_min = np.full((A.shape[0]+(N//2)*2, A.shape[1]+(N//2)*2), 300)
    wall_min[(N//2):wall_min.shape[0]-(N//2), (N//2):wall_min.shape[1]-(N//2)] = A.copy()
    temp_min = np.full((A.shape[0]+(N//2)*2, A.shape[1]+(N//2)*2), 300)
    for y in range(0,wall_min.shape[0]):
        for x in range(0,wall_min.shape[1]):
            if wall_min[y,x]!=300:
                window_min = wall_min[y-(N//2):y+(N//2)+1,x-(N//2):x+(N//2)+1]
                num_min = np.amin(window_min)
                temp_min[y,x] = num_min
    B = temp_min[(N//2):wall_min.shape[0]-(N//2), (N//2):wall_min.shape[1]-(N//2)].copy()
    return B


def background_subtraction(I, B):
    O = I - B
    norm_img = cv2.normalize(O, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    return norm_img


import time
start = time.time()
image = './experiment/5.jpg'
org = cv2.imread(image)
org = cv2.resize(org, (512, 512))
gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
img = max_filtering(100, gray)
back = min_filtering(100, img)
processed = background_subtraction(gray, back)
processed = np.clip(processed, 0, 255)
processed = np.array(processed, np.uint8)
ret3, th3 = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(time.time() - start)

plt.subplot(2, 2, 1), plt.imshow(gray, cmap="gray")
plt.title("org"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(back, cmap="gray")
plt.title("background"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(processed, cmap="gray")
plt.title("norm"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(th3, cmap="gray")
plt.title("result"), plt.xticks([]), plt.yticks([])

plt.show()
