import cv2
import math
import json
import os

import numpy as np

data = []
for file in os.listdir("labels"):
    with open("labels/" + file, "r") as f:
        d = json.load(f)
        points = []
        for elem in d['labels']:
            if elem["label_class"] == "Tray1":
                points = list(map(lambda e: (e['x'], e['y']), elem["regions"][0]))
                break
        data.append({"file": file.split("__")[0] + ".jpg", "points": points})

id = 0


#
# start webcam
def resize(img, width):
    (h, w) = img.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = h / w
    new_height = int(width * aspect_ratio)
    img = cv2.resize(img, (width, new_height))
    return img


def convert_img(img, points):
    lines = np.array(points, dtype=np.int32)
    # Get the original dimensions
    (h, w) = img.shape[:2]
    # Desired width
    new_width = 500

    # Calculate the aspect ratio
    aspect_ratio = h / w
    new_height = int(new_width * aspect_ratio)
    img = cv2.resize(img, (new_width, new_height))

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_img = cv2.GaussianBlur(gray_img, (1, 1), sigmaX=1, sigmaY=1)
    edges = cv2.Laplacian(gray_img, -1)
    # contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # out = img.copy()
    # cv2.drawContours(out, contours, -1, 255, 20)
    # edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # cv2.polylines(edges, [lines], True, (0, 255, 0), 20)


    return edges


for i in data:
    img = cv2.imread("img\\" + i["file"])
    img = convert_img(img, i["points"])
    cv2.imwrite("out/" + i["file"], img)
exit()
while True:
    img = cv2.imread("img\\" + data[id]["file"])

    cv2.imshow('Webcam', convert_img(img, data[id]["points"]))
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('d'):
        id = id + 1
        print(id, data[id]["file"])

cv2.destroyAllWindows()
