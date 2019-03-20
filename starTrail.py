# coding=utf-8
import cv2
import numpy as np
from numba import jit
import os
from collections import Counter
import time


def calcIJ(img_patch):
    total_p = img_patch.shape[0] * img_patch.shape[1]
    center_p = img_patch[img_patch.shape[0] / 2, img_patch.shape[1] / 2]
    mean_p = (np.sum(img_patch) - center_p) / (total_p - 1)
    return (center_p, mean_p)


def calcEntropy2d(img, win_w=3, win_h=3):
    ext_x = win_w / 2
    ext_y = win_h / 2

    # 考虑滑动窗口大小，对原图进行扩边
    final_img = cv2.copyMakeBorder(img, ext_y, ext_y, ext_x, ext_x, cv2.BORDER_REFLECT)

    new_width = final_img.shape[1]
    new_height = final_img.shape[0]

    # 依次获取每个滑动窗口的内容
    IJ = []
    for i in range(img.shape[0] - win_w):
        for j in range(img.shape[1] - win_h):
            patch = final_img[i + ext_x:i + ext_x + win_w + 1, j + ext_y:j + ext_y + win_h + 1]
            IJ.append(calcIJ(patch))

    # 循环遍历统计各二元组个数
    Fij = Counter(IJ).items()

    # 计算各二元组出现的概率
    Pij = []
    for item in Fij:
        Pij.append(item[1] * 1.0 / (new_height * new_width))

    # 计算每个概率所对应的二维熵
    H_tem = []
    for item in Pij:
        h_tem = -item * (np.log(item) / np.log(2))
        H_tem.append(h_tem)

    # 对所有二维熵求和
    H = sum(H_tem)
    return H


def getEntropyMap(img, big_win_w=7, big_win_h=7, small_win_w=3, small_win_h=3):
    if img.shape.__len__() == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ext_x = big_win_w / 2
    ext_y = big_win_h / 2

    final_img = cv2.copyMakeBorder(img, ext_y, ext_y, ext_x, ext_x, cv2.BORDER_REFLECT)
    entropy_img = np.zeros([img.shape[0], img.shape[1]], np.float)

    # 依次获取每个滑动窗口的内容
    total_pixel = img.shape[0] * img.shape[1]
    counter = 0
    times = []
    step = 10000
    for i in range(img.shape[0] - big_win_h):
        for j in range(img.shape[1] - big_win_w):
            patch = final_img[i + ext_x:i + ext_x + big_win_w + 1, j + ext_y:j + ext_y + big_win_h + 1]
            entropy = calcEntropy2d(patch, win_w=small_win_w, win_h=small_win_h)
            entropy_img[i, j] = entropy
            counter += 1
            if counter % step == 0:
                if times.__len__() == 0:
                    times.append(time.time())
                    print round((counter * 1.0 / total_pixel) * 100, 2), "% finished"
                else:
                    times.append(time.time())
                    dt = times[-1] - times[-2]
                    remain_time = round(((total_pixel - counter) / step) * dt)
                    print round((counter * 1.0 / total_pixel) * 100), "% finished,", \
                        "remain time:", remain_time, "s"
    print "100% finished"
    mean_entropy = np.mean(entropy_img)
    entropy_img = np.where(entropy_img == 0, mean_entropy, entropy_img)
    return entropy_img


def getEntropyThreshold(entropy_img, methold='mean'):
    if methold == 'mean':
        mean_entropy = np.mean(entropy_img)
        print "mean", mean_entropy * 1.2
        return mean_entropy
    elif methold == 'stat':
        # 确定合适阈值
        max_val = np.max(entropy_img)
        min_val = np.min(entropy_img)
        print "max", max_val
        print "min", min_val
        step_length = (max_val - min_val) / 10
        bin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(entropy_img.shape[0]):
            for j in range(entropy_img.shape[1]):
                corresponding_index = int((entropy_img[i, j] - min_val) / step_length)
                bin[corresponding_index] += 1
        r1 = bin.index(max(bin))
        threshold = (r1 + 1) * step_length
        return threshold


def getEntropyMask(entropy_img, th):
    mask = np.where(entropy_img > th, 1, 0)
    return mask


def joinPairwiseEntropy(img1, img2, entropy_mask):
    join_img = img1.copy()
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    for i in range(entropy_mask.shape[0]):
        for j in range(entropy_mask.shape[1]):
            if entropy_mask[i, j] == 1:
                # 对于非地物，取最小值
                if img2_gray[i, j] < img1_gray[i, j]:
                    join_img[i, j] = img2[i, j]
            else:
                # 对于星空，取最大值
                if img2_gray[i, j] > img1_gray[i, j]:
                    join_img[i, j] = img2[i, j]
    return join_img


def joinBatchEntropy(imgs, entropy_reference=0):
    print "Initializing ..."
    img1 = cv2.imread(imgs[0])
    img2 = cv2.imread(imgs[1])

    entropy_img = getEntropyMap(cv2.imread(imgs[entropy_reference]))
    entropy_th = getEntropyThreshold(entropy_img, methold='mean')
    mask = getEntropyMask(entropy_img, entropy_th)

    print "starting using entropy method for pixel fusion..."
    join = joinPairwiseEntropy(img1, img2, mask)
    for i in range(2, imgs.__len__()):
        print i + 1, "/", imgs.__len__()
        tmp_img = cv2.imread(imgs[i])
        join = joinPairwiseEntropy(join, tmp_img, mask)
    return join


def findAllFiles(root_dir, filter):
    """
    在指定目录查找指定类型文件

    :param root_dir: 查找目录
    :param filter: 文件类型
    :return: 路径、名称、文件全路径

    """

    print("Finding files ends with \'" + filter + "\' ...")
    separator = os.path.sep
    paths = []
    names = []
    files = []
    for parent, dirname, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(filter):
                paths.append(parent + separator)
                names.append(filename)
    for i in range(paths.__len__()):
        files.append(paths[i] + names[i])
    print (names.__len__().__str__() + " files have been found.")
    paths.sort()
    names.sort()
    files.sort()
    return paths, names, files


def joinPairwiseTh(img1, img2, diffTh=10):
    join_img = img1.copy()
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = img2_gray.astype(np.int) - img1_gray.astype(np.int)
    diff_pixels = np.argwhere(diff > diffTh)
    for i in range(diff_pixels.__len__()):
        join_img[diff_pixels[i][0], diff_pixels[i][1]] = img2[diff_pixels[i][0], diff_pixels[i][1]]
    return join_img


def joinPairwiseMax(img1, img2):
    join_img = img1.copy()
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = img2_gray.astype(np.int) - img1_gray.astype(np.int)
    diff_pixels = np.argwhere(diff >= 0)
    for i in range(diff_pixels.__len__()):
        join_img[diff_pixels[i][0], diff_pixels[i][1]] = img2[diff_pixels[i][0], diff_pixels[i][1]]
    return join_img


def joinPairwiseMin(img1, img2):
    join_img = img1.copy()
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = img2_gray.astype(np.int) - img1_gray.astype(np.int)
    diff_pixels = np.argwhere(diff <= 0)
    for i in range(diff_pixels.__len__()):
        join_img[diff_pixels[i][0], diff_pixels[i][1]] = img2[diff_pixels[i][0], diff_pixels[i][1]]
    return join_img


@jit(nopython=True)
def joinPairwiseMean(img1, img2):
    join_img = img1.copy()
    for i in range(img1.shape[0]):
        for j in range(img2.shape[1]):
            mean_b = (img1[i, j, 0] + img2[i, j, 0]) / 2
            mean_g = (img1[i, j, 1] + img2[i, j, 1]) / 2
            mean_r = (img1[i, j, 2] + img2[i, j, 2]) / 2
            join_img[i, j] = [mean_b, mean_g, mean_r]
    return join_img


def joinBatch(imgs, method='max', diffTh=10):
    print "Initializing ..."
    img1 = cv2.imread(imgs[0])
    img2 = cv2.imread(imgs[1])
    if method is 'max':
        print "starting using max method for pixel fusion..."
        join = joinPairwiseMax(img1, img2)
        for i in range(2, imgs.__len__()):
            print i + 1, "/", imgs.__len__()
            tmp_img = cv2.imread(imgs[i])
            join = joinPairwiseMax(join, tmp_img)
    elif method is 'min':
        print "starting using min method for pixel fusion..."
        join = joinPairwiseMin(img1, img2)
        for i in range(2, imgs.__len__()):
            print i + 1, "/", imgs.__len__()
            tmp_img = cv2.imread(imgs[i])
            join = joinPairwiseMin(join, tmp_img)
    elif method is 'mean':
        print "starting using mean method for pixel fusion..."
        join = joinPairwiseMean(img1, img2)
        for i in range(2, imgs.__len__()):
            print i + 1, "/", imgs.__len__()
            tmp_img = cv2.imread(imgs[i])
            join = joinPairwiseMean(join, tmp_img)
    elif method is 'th':
        print "starting using threshold method for pixel fusion..."
        join = joinPairwiseTh(img1, img2, diffTh=diffTh)
        for i in range(2, imgs.__len__()):
            print i + 1, "/", imgs.__len__()
            tmp_img = cv2.imread(imgs[i])
            join = joinPairwiseTh(join, tmp_img, diffTh=diffTh)

    print "star trail join finished!"
    return join


if __name__ == '__main__':
    _, _, fullnames = findAllFiles("./test/", ".jpg")
    res1 = joinBatchEntropy(fullnames)
    res2 = joinBatch(fullnames)
    cv2.imwrite("res1.jpg", res1)
    cv2.imwrite("res2.jpg", res2)
