import cv2
import math as mt
import numpy as np
import matplotlib.pyplot as plt


def onclick_1(event):
    print("onclick")
    x = event.xdata
    y = event.ydata
    plt.plot(x, y, '*r')
    plt.show()
    if len(locations_1) >= 4:
        locations_2.append([int(x), int(y)])
    else:
        locations_1.append([int(x), int(y)])


# def onclick_2(event):
#     global locations_2, fig2
#     print("onclick_2")
#     x = event.xdata
#     y = event.ydata
#     plt.plot(x, y, '*r')
#     plt.show()
#     locations_2.append([x, y])


# Q1.1
def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """
    Find the disparity matrix that represent the differences of positions between left image and right image.
    disp_map[i][j] = J_R - J_L

    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)
    return: Disparity map, disp_map.shape = Left.shape
    """
    disp_map = np.zeros(img_l.shape)
    kernel = np.zeros((k_size*2 + 1, k_size*2 + 1))
    krows = kernel.shape[0]
    kcols = kernel.shape[1]
    for r in range((krows-1)//2, img_l.shape[0]-((krows-1)//2)):
        for c in range((kcols-1)//2, img_r.shape[1]-((kcols-1)//2)):
            x, y = findBestMatchingSSD(img_l, img_r, r, c, krows, kcols)
            disp_map[r][c] = abs(y-c)/256
    print(disp_map)
    return disp_map


def findBestMatchingSSD(img_l, img_r, r, c, krows, kcols):
    """
    Find the best matching in right scan-line. We assume the correspondence between the images rows,
    therefore we examine just the matching row in right image.
    :param img_l:
    :param img_r:
    :param r:
    :param c:
    :param krows:
    :param kcols:
    :return: xmin, ymin (min index)
    """
    xmin = 0
    ymin = 0
    _min = 0
    for col in range((kcols-1)//2, img_r.shape[1]-((kcols-1)//2)):
        _sum = 0
        for i in range(krows):
            for j in range(kcols):
                _sum += (img_l[r - ((krows-1)//2) + i][c - ((kcols-1)//2) + j] -
                         img_r[r - ((krows-1)//2) + i][col - ((kcols-1)//2) + j])**2
        if _sum < _min:
            _min = _sum
            xmin = r
            ymin = col
    return xmin, ymin


# Q1.2
def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)
    return: Disparity map, disp_map.shape = Left.shape
    """
    disp_map = np.zeros(img_l.shape)
    kernel = np.zeros((k_size * 2 + 1, k_size * 2 + 1))
    krows = kernel.shape[0]
    kcols = kernel.shape[1]
    for r in range((krows-1)//2, img_l.shape[0]-((krows-1)//2)):
        for c in range((kcols-1)//2, img_r.shape[1]-((kcols-1)//2)):
            x, y = findBestMatchingNC(img_l, img_r, r, c, krows, kcols)
            disp_map[r][c] = abs(y - c) / 256
    print(disp_map)
    return disp_map


def findBestMatchingNC(img_l, img_r, r, c, krows, kcols):
    _max = 0
    xmax = 0
    ymax = 0
    for col in range((kcols-1)//2, img_r.shape[1]-((kcols-1)//2)):
        _sum1 = 0
        _sum2 = 0
        _sum3 = 0
        tsum = 0
        for i in range(krows):
            for j in range(kcols):
                _sum1 += (img_l[r - ((krows-1)//2) + i][c - ((kcols-1)//2) + j] *
                          img_r[r - ((krows-1)//2) + i][col - ((kcols-1)//2) + j])
                _sum2 += (img_r[r - ((krows-1)//2) + i][c - ((kcols-1)//2) + j] *
                          img_r[r - ((krows-1)//2) + i][col - ((kcols-1)//2) + j])
                _sum3 += (img_l[r - ((krows-1)//2) + i][c - ((kcols-1)//2) + j] *
                          img_l[r - ((krows-1)//2) + i][col - ((kcols-1)//2) + j])
        tsum = _sum1/mt.sqrt(_sum2*_sum3)
        if tsum > _max:
            _max = tsum
            xmax = r
            ymax = col
    return xmax, ymax


# Q2.1
def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points.
    Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))
    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]
    return: (Homography matrix shape:[3,3], Homography error)
    """
    M = np.zeros((2*src_pnt.shape[0], 9))
    i = 0
    while i < 8:
        M[i][0] = src_pnt[i//2][0]
        M[i][1] = src_pnt[i//2][1]
        M[i][2] = 1
        M[i][6] = -1*(dst_pnt[i//2][0]*src_pnt[i//2][0])
        M[i][7] = -1*(dst_pnt[i//2][0]*src_pnt[i//2][1])
        M[i][8] = -1*(dst_pnt[i//2][0])
        M[i+1][3] = src_pnt[i//2][0]
        M[i+1][4] = src_pnt[i//2][1]
        M[i+1][5] = 1
        M[i+1][6] = -1*(dst_pnt[i//2][1]*src_pnt[i//2][0])
        M[i+1][7] = -1*(dst_pnt[i//2][1]*src_pnt[i//2][1])
        M[i+1][8] = -1*(dst_pnt[i//2][1])
        i += 2
    print(M)
    # M = np.dot(M.T, M)
    U, D, V = np.linalg.svd(M)
    print("D = ", D)
    V = np.dot(V.T, V)
    print("V = ", V)
    H = V[:, V.shape[1]-1]
    H = np.reshape(H, (3, 3))
    print("H = ", H)
    # =========== OpenCV - Homography ===========
    M, mask = cv2.findHomography(src_pnt, dst_pnt)
    print("opencv = ", M)


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image. Then calculates the homography and transforms the source image on to the destination image. Then transforms the source image onto the destination image and displays the result.
    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.
    output:
    None.
    """
    global fig1, fig2, locations_1, locations_2
    locations_1 = []
    locations_2 = []
    # display image 1
    fig1 = plt.figure()
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(src_img)
    plt.show()
    print(locations_1)

    # display image 2
    fig2 = plt.figure()
    cid = fig2.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    print(locations_2)

    # compute Homography
    srcPoints = np.copy(locations_1)
    dstPoints = np.copy(locations_2)
    H, mask = cv2.findHomography(srcPoints, dstPoints)

    # warp the images
    # H = H.astype('float32')
    new_image = warpImages(src_img, dst_img, H)


def warpImages(img_, img, H):
    """
    :param img_: right image
    :param img: left image
    :param H: homography matrix
    :return:
    """
    # OpenCV warping
    img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)  # right gray
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # left gray
    M = H
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    #
    dst = cv2.warpPerspective(img_, M, (img.shape[1] + img_.shape[1], img.shape[0]))
    dst[0:img.shape[0], 0:img.shape[1]] = img
    cv2.imshow('fig', dst)
    cv2.waitKey(0)
    return dst





im1 = cv2.imread('pair1-L.png')
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
# cv2.imshow('pair1-L', im1)
# cv2.waitKey(0)
# im2 = cv2.imread('pair1-R.png')
# im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
# cv2.imshow('pair1-R', im2)
# cv2.waitKey(0)
# disp1 = disparitySSD(im1, im2, 0, 1)
# cv2.imshow('disparity-map', disp1)
# cv2.waitKey(0)
# disp2 = disparityNC(im1, im2, 0, 1)
# cv2.imshow('disparity-map', disp2)
# cv2.waitKey(0)
src_pnt = np.array([[279, 552],
                    [372, 559],
                    [362, 472],
                    [277, 469]])
dst_pnt = np.array([[24, 566],
                    [114, 552],
                    [106, 474],
                    [19, 481]])
# computeHomography(src_pnt, dst_pnt)


src = cv2.imread('pair2-L.png')
# src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
dst = cv2.imread('pair2-R.png')
# dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
warpImag(src, dst)


