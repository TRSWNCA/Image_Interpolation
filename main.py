import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter
from math import sqrt, floor, ceil
from PIL import Image
import math


UpperVal = 240

def check(orig, x, y):
    if int(orig[x][y][0]) + int(orig[x][y][1]) + int(orig[x][y][2]) >= 3 * UpperVal:
        return True
    return False


# Mean square error
def MSE(imgA, imgB):
    rows, cols = imgA.shape
    sumimg = 0
    for i in range(rows):
        for j in range(cols):
            sumimg = sumimg + ((float(imgA[i, j]) - float(imgB[i, j])) ** 2)
    return sumimg / (rows * cols)


# Peak signal to noise ratio
def PSNR(imgA, imgB):
    rows, cols = imgA.shape
    sumimg = 0
    for i in range(rows):
        for j in range(cols):
            sumimg = sumimg + ((float(imgA[i, j]) - float(imgB[i, j])) ** 2)
    mse = sumimg / (rows * cols)
    const = 255 ** 2
    frac = float(const / mse)
    return 10 * np.log10(frac)

#SSIM Structural SIMilarity Index (SSIM).
def ssim(imgA, imgB):
    def __get_kernels():
        k1, k2, l = (0.01, 0.03, 255.0)
        kern1, kern2 = map(lambda x: (x * l) ** 2, (k1, k2))
        return kern1, kern2

    def __get_mus(i1, i2):
        mu1, mu2 = map(lambda x: gaussian_filter(x, 1.5), (i1, i2))
        m1m1, m2m2, m1m2 = (mu1 * mu1, mu2 * mu2, mu1 * mu2)
        return m1m1, m2m2, m1m2

    def __get_sigmas(i1, i2, delta1, delta2, delta12):
        f1 = gaussian_filter(i1 * i1, 1.5) - delta1
        f2 = gaussian_filter(i2 * i2, 1.5) - delta2
        f12 = gaussian_filter(i1 * i2, 1.5) - delta12
        return f1, f2, f12

    def __get_positive_ssimap(C1, C2, m1m2, mu11, mu22, s12, s1s1, s2s2):
        num = (2 * m1m2 + C1) * (2 * s12 + C2)
        den = (mu11 + mu22 + C1) * (s1s1 + s2s2 + C2)
        return num / den

    def __get_negative_ssimap(C1, C2, m1m2, m11, m22, s12, s1s1, s2s2):
        (num1, num2) = (2.0 * m1m2 + C1, 2.0 * s12 + C2)
        (den1, den2) = (m11 + m22 + C1, s1s1 + s2s2 + C2)
        ssim_map = __n.ones(img1.shape)
        indx = (den1 * den2 > 0)
        ssim_map[indx] = (num1[indx] * num2[indx]) / (den1[indx] * den2[indx])
        indx = __n.bitwise_and(den1 != 0, den2 == 0)
        ssim_map[indx] = num1[indx] / den1[indx]
        return ssim_map

    (img1, img2) = (imgA.astype('double'), imgB.astype('double'))
    (m1m1, m2m2, m1m2) = __get_mus(img1, img2)
    (s1, s2, s12) = __get_sigmas(img1, img2, m1m1, m2m2, m1m2)
    (C1, C2) = __get_kernels()
    if C1 > 0 and C2 > 0:
        ssim_map = __get_positive_ssimap(C1, C2, m1m2, m1m1, m2m2, s12, s1, s2)
    else:
        ssim_map = __get_negative_ssimap(C1, C2, m1m2, m1m1, m2m2, s12, s1, s2)
    ssim_value = ssim_map.mean()
    return ssim_value

def nearest_interpolation(orig):
    img = orig.copy()
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if int(img[x][y][0]) + int(img[x][y][1]) + int(img[x][y][2]) >= 3 * UpperVal:
                result = []
                for i in range(img.shape[2]):
                    q = [(x, y)]
                    l = 0
                    nx, ny = x, y
                    while 1:
                        nx, ny = q[l]
                        l += 1
                        if not check(img, nx, ny):
                            break
                        q.append((x - 1, y))
                        q.append((x, y - 1))
                    a = int(img[nx][ny][i])
                    result.append(a)
                img[x][y][0], img[x][y][1], img[x][y][2] = result[0], result[1], result[2]
    return img


def bilinear_interpolation(orig):
    img = orig.copy()
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if int(img[x][y][0]) + int(img[x][y][1]) + int(img[x][y][2]) >= 3 * UpperVal:
                result = []
                for i in range(img.shape[2]):
                    a = int(img[x - 5][y + 5][i])
                    b = int(img[x - 5][y][i])
                    c = int(img[x][y - 5][i])
                    d = int(img[x - 5][y - 5][i])
                    ans = int(7 / 23 * d + 6 / 23 * (b + c) + 4 / 23 * a)
                    result.append(ans)
                img[x][y][0], img[x][y][1], img[x][y][2] = result[0], result[1], result[2]
    return img

def rbf(orig, f):
    img = orig.copy()
    last, now = "", ""
    for i in range(img.shape[2]):
        for x in range(img.shape[0]):
            last = now
            now = "%.2f" % float(x / img.shape[0] / 3 + i * 0.33)
            for y in range(img.shape[1]):
                if int(orig[x][y][0]) + int(orig[x][y][1]) + int(orig[x][y][2]) >= 3 * UpperVal:
                    known_x, known_y = [], []
                    for y2 in range(y - 20, y + 20):
                        if int(orig[x][y2][0]) + int(orig[x][y2][1]) + int(orig[x][y2][2]) < 3 * UpperVal:
                            known_x.append(y2)
                            known_y.append(int(img[x][y2][i]))
                    if len(known_x) < 2:
                        continue
                    rf = Rbf(known_x, known_y, function=f)
                    predict = rf([y])
                    predict = np.array(predict).astype(np.uint8)
                    img[x][y][i] = predict[0]
    return img


def solve(header, imgA, imgB):
    imgA = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
    imgB = cv.cvtColor(imgB, cv.COLOR_BGR2GRAY)
    print(header, MSE(imgA, imgB), PSNR(imgA, imgB), ssim(imgA, imgB))


def main():
    img = cv.imread('data/words/1.bmp')
    print("bilinear begin.")
    b_result = bilinear_interpolation(img)
    print("bilinear done.")
    print("nearest begin.")
    n_result = nearest_interpolation(img)
    print("nearest done.")
    print("rbf linear begin.")
    r1_result = rbf(img, 'linear')
    print("rbf linear down.")
    print("rbf gaussian begin.")
    r2_result = rbf(img, 'gaussian')
    print("rbf gaussian down.")
    print("rbf inverse begin.")
    r3_result = rbf(img, 'inverse')
    print("rbf inverse down.")

    print("        MSE  PSNR  ssim")
    solve('bilinear', img, b_result)
    solve('nearest', img, n_result)
    solve('rbf 1', img, r1_result)
    solve('rbf 2', img, r2_result)
    solve('rbf 3', img, r3_result)

    plt.figure()

    plt.subplot(3, 2, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('origin')
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.imshow(cv.cvtColor(b_result, cv.COLOR_BGR2RGB))
    plt.title('bilinear')
    plt.axis('off')

    plt.subplot(3, 2, 3)
    plt.imshow(cv.cvtColor(n_result, cv.COLOR_BGR2RGB))
    plt.title('nearest')
    plt.axis('off')

    plt.subplot(3, 2, 4)
    plt.imshow(cv.cvtColor(r1_result, cv.COLOR_BGR2RGB))
    plt.title('rbf linear')
    plt.axis('off')

    plt.subplot(3, 2, 5)
    plt.imshow(cv.cvtColor(r2_result, cv.COLOR_BGR2RGB))
    plt.title('rbf gaussian')
    plt.axis('off')

    plt.subplot(3, 2, 6)
    plt.imshow(cv.cvtColor(r3_result, cv.COLOR_BGR2RGB))
    plt.title('rbf inverse')

    plt.tight_layout()
    plt.axis('off')
    plt.show()

    cv.imwrite('bilinear.jpg', b_result)
    cv.imwrite('nearest.jpg', n_result)
    cv.imwrite('rbf1.jpg', r1_result) #np.hstack((img, result)))
    cv.imwrite('rbf2.jpg', r2_result) #np.hstack((img, result)))
    cv.imwrite('rbf3.jpg', r3_result) #np.hstack((img, result)))


if __name__ == '__main__':
    main()
