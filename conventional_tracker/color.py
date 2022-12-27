import cv2
import numpy as np
from PIL import Image as im


def max_color(frame, box):

    data = im.fromarray(frame)

    (x, y, w, h, _) = box
    im_trim1 = data.crop((x, y, x+w, y+h))
    imgn = np.asarray(im_trim1)

    height, width, _ = np.shape(imgn)
    # print(height, width)

    data2 = np.reshape(imgn, (height * width, 3))
    data2 = np.float32(data2)

    #number_clusters = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(
        data2, 1, None, criteria, 10, flags)
    centers = np.around(centers)
    centers = centers.tolist()
    return centers
