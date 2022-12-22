import random
import threading

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter
from math import sqrt, floor, ceil
from PIL import Image
import math


all_missed = []
ans = {}

# img = None

dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

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


def rbf_point(f, x, y, i):
    known_x, known_y, known_d = [], [], []
    for x2 in range(max(x - 10, 0), min(x + 10, img.shape[0])):
        for y2 in range(max(y - 10, 0), min(y + 10, img.shape[1])):
            if not check(img, x2, y2):
                if not int(img[x2][y2][i]) in known_d:
                    known_x.append(x2)
                    known_y.append(y2)
                    known_d.append(int(img[x2][y2][i]))
    if len(known_x) == 1:
        ans[(x, y, i)] = known_d[0]
        return
    rf = Rbf(known_x, known_y, known_d, function=f)
    predict = rf([x], [y])
    predict = np.array(predict).astype(np.uint8)
    ans[(x, y, i)] = predict[0]

    # while 1:
    #     x2, y2 = q[l]
    #     v[x2][y2] = 1
    #     l += 1
    #     if not check(img, x2, y2):
    #         if not int(img[x2][y2][i]) in known_d:
    #             known_x.append(x2)
    #             known_y.append(y2)
    #             known_d.append(int(img[x2][y2][i]))
    #     for w in range(4):
    #         nx, ny = x2 + dx[w], y2 + dy[w]
    #         if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
    #             if not v[nx][ny]:
    #                 q.append((nx, ny))
    #     if len(known_x) > 2:
    #         break
    # rf = Rbf(known_x, known_y, known_d, function=f)
    # predict = rf([x], [y])
    # predict = np.array(predict).astype(np.uint8)
    # ans[(x, y, i)] = predict[0]

cnt = 0

def rbf_solo(id, f):
    for k in range(3):
        for i in range(id, len(all_missed), 50):
            x, y = all_missed[i]
            rbf_point(f, x, y, k)
            global cnt
            cnt += 1
            print(cnt / 3 / len(all_missed))


def rbf_par(f):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if check(img, x, y):
                all_missed.append((x, y))
    threads = []
    for i in range(50):
        threads.append(threading.Thread(target=rbf_solo, args=(i, f)))
        threads[i].start()

    for p in threads:
        p.join()
    for (x, y) in all_missed:
        img[x][y][0] = ans[(x, y, 0)]
        img[x][y][1] = ans[(x, y, 1)]
        img[x][y][2] = ans[(x, y, 2)]


def rbf_old(orig, f):
    img = orig.copy()
    last, now = '', ''
    for i in range(img.shape[2]):
        for x in range(img.shape[0]):
            last = now
            now = "%.2f" % float(x / img.shape[0] / 3 + i * 0.33)
            # if last != now:
            #     print(now)
            for y in range(img.shape[1]):
                if check(orig, x, y):
                    known_x, known_y, known_d = [], [], []
                    for x2 in range(max(x - 2, 0), min(x + 2, img.shape[0])):
                        for y2 in range(max(y - 10, 0), min(y + 10, img.shape[1])):
                            if not check(img, x2, y2):
                                if not int(img[x2][y2][i]) in known_d:
                                    known_x.append(x2)
                                    known_y.append(y2)
                                    known_d.append(int(img[x2][y2][i]))
                    if len(known_x) == 1:
                        img[x][y][i] = known_d[0]
                        continue
                    if len(known_x) == 0:
                        print(x, y)
                        continue
                    rf = Rbf(known_x, known_y, known_d, function=f)
                    predict = rf([x], [y])
                    predict = np.array(predict).astype(np.uint8)
                    img[x][y][i] = predict[0]
    return img

def rbf(orig, f):
    img = orig.copy()
    sz = 10
    for i in range(img.shape[0] // sz):
        lx = i * sz
        rx = min(img.shape[0], (i + 1) * sz)
        for j in range(img.shape[1] // sz):
            ly = j * sz
            ry = min(img.shape[1], (j + 1) * sz)
            for k in range(3):
                known_x, known_y, known_d = [], [], []
                miss_x, miss_y = [], []
                for x in range(lx, rx):
                    for y in range(ly, ry):
                        if check(orig, x, y):
                            miss_x.append(x)
                            miss_y.append(y)
                        else:
                            if not int(img[x][y][k]) in known_d:
                                known_x.append(x)
                                known_y.append(y)
                                known_d.append(int(img[x][y][k]))
                if len(known_x) == 1:
                    known_x.append(known_x[0] - 1)
                    known_y.append(known_y[0] - 1)
                    known_d.append(known_d[0] - 1)
                rf = Rbf(known_x, known_y, known_d, function=f)
                predict = rf(miss_x, miss_y)
                predict = np.array(predict).astype(np.uint8)
                for (index, val) in enumerate(predict):
                    img[miss_x[index]][miss_y[index]][k] = predict[index]
    return img

def rbf_task1(orig, f):
    img = orig.copy()
    last, now = "", ""
    for i in range(img.shape[2]):
        for x in range(img.shape[0]):
            last = now
            now = "%.2f" % float(x / img.shape[0] / 3 + i * 0.33)
            for y in range(img.shape[1]):
                if int(orig[x][y][0]) + int(orig[x][y][1]) + int(orig[x][y][2]) >= 3 * UpperVal:
                    known_x, known_y = [], []
                    for y2 in range(max(0, y - 20), min(img.shape[1], y + 20)):
                        if int(orig[x][y2][0]) + int(orig[x][y2][1]) + int(orig[x][y2][2]) < 3 * UpperVal:
                            known_x.append(y2)
                            known_y.append(int(img[x][y2][i]))
                    if len(known_x) == 1:
                        img[x][y][i] = known_y[0]
                        continue
                    if len(known_x) == 0:
                        img[x][y][i] = img[max(0, x - 1)][y][i]
                        continue
                    rf = Rbf(known_x, known_y, function=f)
                    predict = rf([y])
                    predict = np.array(predict).astype(np.uint8)
                    img[x][y][i] = predict[0]
    return img

def solve(imgA, imgB):
    imgA = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
    imgB = cv.cvtColor(imgB, cv.COLOR_BGR2GRAY)
    print(MSE(imgA, imgB), PSNR(imgA, imgB), ssim(imgA, imgB))


def miss_gen():
    for id in range(1, 4):
        img = cv.imread('data/words/%d.jpg' % id)
        for i in range(10, 100, 10):
            new_img = img.copy()
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    val = random.randint(0, 100)
                    if val < i:
                        new_img[x][y] = [255, 255, 255]
            cv.imwrite('data/miss/%d_%d.bmp' % (id, i), new_img)

def main():
    for way in ['inverse']:
        for id in range(1, 4):
            plt.figure()
            # plt.subplot(3, 3, 1)
            # orig = cv.imread('data/words/%d.jpg' % id)
            # plt.axis('off')
            # plt.title('origin')
            # plt.imshow(cv.cvtColor(orig, cv.COLOR_BGR2RGB))
            print(way, id)
            for prob in range(10, 100, 10):
                global img
                img = cv.imread('data/miss/%d_%d.bmp' % (id, prob))
                # r1 = rbf_task1(img, 'linear')
                r1 = rbf_old(img, way)
                solve(img, r1)
                cv.imwrite('data/result/%s_%d_%d.bmp' % (way, id, prob), r1)
                plt.subplot(3, 3, prob // 10)
                plt.imshow(cv.cvtColor(r1, cv.COLOR_BGR2RGB))
                plt.title(str(prob) + "%")
                plt.axis('off')
            plt.savefig('%d_%s.png' % (id, way), dpi=100)




if __name__ == '__main__':
    main()
