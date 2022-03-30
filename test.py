import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


def hanlde_img(path):
    # 根据路径读取图片
    img = cv2.imread(path)

    img = img.astype(np.float) / 255.0

    # 分离 RGB 三个通道，注意：openCV 中图像格式是 BGR
    srcR = img[:, :, 2]
    srcG = img[:, :, 1]
    srcB = img[:, :, 0]

    # 将原图转成灰度图
    grayImg = 0.299 * srcR + 0.587 * srcG + 0.114 * srcB

    # 高光选区
    # maskThreshold = 0.64
    # luminance = grayImg * grayImg
    # luminance = np.where(luminance > maskThreshold, luminance, 0)

    # 阴影选区
    maskThreshold = 0.33
    luminance = (1 - grayImg) * (1 - grayImg)
    luminance = np.where(luminance > maskThreshold, luminance, 0)

    mask = luminance > maskThreshold

    # 显示正交叠底图
    # img[:, :, 0] = luminance
    # img[:, :, 1] = luminance
    # img[:, :, 2] = luminance

    # 显示选区内原图
    img[:, :, 0][~mask] = 0
    img[:, :, 1][~mask] = 0
    img[:, :, 2][~mask] = 0

    img = img * 255
    img = img.astype(np.uint8)

    # 创建图片显示窗口
    title = "ShadowHighlight"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 800, 600)
    cv2.moveWindow(title, 0, 0)
    while True:
        # 循环显示图片，按 ‘q’ 键退出
        cv2.imshow(title, img)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()


class PSShadowHighlight:
    """
    色阶调整
    """

    def __init__(self, image):
        self.shadows_light = 50

        img = image.astype(np.float) / 255.0

        srcR = img[:, :, 2]
        srcG = img[:, :, 1]
        srcB = img[:, :, 0]
        srcGray = 0.299 * srcR + 0.587 * srcG + 0.114 * srcB

        # 阴影选区
        luminance = (1 - srcGray) * (1 - srcGray)

        ## 高光选区
        #luminance = luminance * luminance
        #luminance = np.where(luminance > 0.64, luminance, 0)

        self.maskThreshold = np.mean(luminance)
        mask = luminance > self.maskThreshold

        imgRow = np.size(img, 0)
        imgCol = np.size(img, 1)
        print("imgRow:%d, imgCol:%d, maskThreshold:%f" % (imgRow, imgCol, self.maskThreshold))
        print("shape:", img.shape)

        self.rgbMask = np.zeros([imgRow, imgCol, 3], dtype=bool)
        self.rgbMask[:, :, 0] = self.rgbMask[:, :, 1] = self.rgbMask[:, :, 2] = mask

        self.rgbLuminance = np.zeros([imgRow, imgCol, 3], dtype=float)
        self.rgbLuminance[:, :, 0] = self.rgbLuminance[:, :, 1] = self.rgbLuminance[:, :, 2] = luminance

        self.midtonesRate = np.zeros([imgRow, imgCol, 3], dtype=float)
        self.brightnessRate = np.zeros([imgRow, imgCol, 3], dtype=float)

    def adjust_image(self, img):
        maxRate = 4
        brightness = (self.shadows_light / 100.0 - 0.0001) / maxRate
        midtones = 1 + maxRate * brightness

        self.midtonesRate[self.rgbMask] = midtones
        self.midtonesRate[~self.rgbMask] = (midtones - 1.0) / self.maskThreshold * self.rgbLuminance[
            ~self.rgbMask] + 1.0

        self.brightnessRate[self.rgbMask] = brightness
        self.brightnessRate[~self.rgbMask] = (1 / self.maskThreshold * self.rgbLuminance[~self.rgbMask]) * brightness

        outImg = 255 * np.power(img / 255.0, 1.0 / self.midtonesRate) * (1.0 / (1 - self.brightnessRate))

        img = outImg
        img[img < 0] = 0
        img[img > 255] = 255

        img = img.astype(np.uint8)
        return img


def ps_shadow_highlight_adjust_and_save_img(psSH, origin_image):
    psSH.shadows_light = 50
    image = psSH.adjust_image(origin_image)
    cv2.imwrite('py_sh_out_01.png', image)


def ps_shadow_highlight_adjust(path):
    """
    阴影提亮调整
    """
    origin_image = cv2.imread(path)

    psSH = PSShadowHighlight(origin_image)

    # ps_shadow_highlight_adjust_and_save_img(psSH, origin_image)

    def update_shadows_light(x):
        psSH.shadows_light = x

    # 创建图片显示窗口
    title = "ShadowHighlight"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 800, 600)
    cv2.moveWindow(title, 0, 0)

    # 创建阴影提亮操作窗口
    option_title = "Option"
    cv2.namedWindow(option_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(option_title, 400, 20)
    cv2.moveWindow(option_title, 0, 630)
    cv2.createTrackbar('shadows_light', option_title, psSH.shadows_light, 100, update_shadows_light)

    while True:
        image = psSH.adjust_image(origin_image)
        cv2.imshow(title, image)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()


def unevenLightCompensate(gray, blockSize):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)
    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))
    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]
            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)

            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst[dst > 255] = 255
    dst[dst < 0] = 0
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    # dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return dst


def reduce_highlights(img, highlight_threshold=127):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 先轉成灰階處理
    ret, thresh = cv2.threshold(img_gray, highlight_threshold, 255, 0)  # 利用 threshold 過濾出高光的部分，目前設定高於 200 即為高光
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_zero = np.zeros(img.shape, dtype=np.uint8)

    #     print(len(contours))

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        img_zero[y:y + h, x:x + w] = 255
        mask = img_zero

    print("Highlight part: ")

    # alpha，beta 共同決定高光消除後的模糊程度
    # alpha: 亮度的缩放因子，默認是 0.2， 範圍[0, 2], 值越大，亮度越低
    # beta:  亮度缩放後加上的参数，默認是 0.4， 範圍[0, 2]，值越大，亮度越低
    result = cv2.illuminationChange(img, mask, alpha=0.2, beta=0.2)
    #     show_img(result)

    return result


if __name__ == '__main__':
    '''
        运行环境：Python 3
        执行：python3 py_pic_handle.py <图片路径>
        如：python3 py_pic_handle.py test.jpg
    '''
    from matplotlib import pyplot as plt
    img_path = './experiment/5.jpg'
    #hanlde_img(img_path)
    ps_shadow_highlight_adjust(img_path)
    #blockSize = 8
    #img = cv2.imread(img_path)
    #result = reduce_highlights(img)
    #b, g, r = cv2.split(img)
    #dstb = unevenLightCompensate(b, blockSize)
    #dstg = unevenLightCompensate(g, blockSize)
    #dstr = unevenLightCompensate(r, blockSize)
    #dst = cv2.merge([dstb, dstg, dstr])
    #result = np.concatenate([img, dst], axis=1)
    #plt.imshow(img)
    #plt.show()
