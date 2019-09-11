import cv2
import numpy as np
# help(cv2.xfeatures2d)


img_ = cv2.imread('pair2-R.png')
img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
img = cv2.imread('pair2-L.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ==================================
# ==========find key points=========
# ==================================


# ===========compute homograpy and images stitching
src_pts = np.float32([[139, 430], [617, 422], [615, 241], [79, 244]]).reshape(-1,1,2)
dst_pts = np.float32([[59, 429], [551, 430], [555, 240], [24, 243]]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts)
h,w = img1.shape
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, M)

# need find index that in the negative side of the image(-89.1) #

print(dst)

dst = cv2.warpPerspective(img_, M,(img.shape[1] + img_.shape[1], img.shape[0]))
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = np.pad(dst, ((89, 0), (89, 0)), 'constant')
print(dst.shape)
for i in range(dst.shape[0]):
    for j in range(dst.shape[1]):
        if i >= img.shape[0]-1 or j >= img.shape[1]-1:
            break
        else:
            dst[88+i][j] = img[i][j]
# dst[0:img.shape[0],0:img.shape[1]] = img
cv2.imshow("original_image_stitched.jpg", dst)
cv2.waitKey(0)


def trim(frame):
    # crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    # crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    # crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    # crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame


cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
cv2.waitKey(0)
