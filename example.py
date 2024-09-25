import numpy as np
import os

import tensorflow as tf
import cv2

img_height = 256
img_width = 256
model = tf.keras.models.load_model('out_model')


def find_points(img):
    (h, w) = img.shape[:2]
    input_img = cv2.resize(img, (img_width, img_height))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    input_img = cv2.Laplacian(input_img, -1)
    input_img = input_img.astype(np.float32)
    input_img /= 255.
    out = list(model.predict(np.array([input_img])))[0]
    cvt_point = []
    i = 0
    while i < len(out):
        cvt_point.append((out[i] * w, out[i + 1] * h))
        i += 2
    cvt_point = np.array(cvt_point, dtype=np.int32)
    return cvt_point

for file in os.listdir("img"):
    img = cv2.imread("img/" + file)
    points = find_points(img)
    print(points)
    cv2.polylines(img, [points], True, 255, 3)
    cv2.imwrite("result/" + str(file) + ".jpg", img)
