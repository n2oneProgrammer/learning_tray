import numpy as np
import os
import PIL
import PIL.Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

batch_size = 32
img_height = 256
img_width = 256

x = []
y = []

model = tf.keras.models.load_model('out_model')

new_x = []
for file in os.listdir("test/images"):
    im = cv2.imread("test/images/" + file)
    im = cv2.resize(im, (img_width, img_height))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(np.float32)
    im /= 255.
    new_x.append(im)
new_x = np.array(new_x)
out = list(model.predict(new_x))

for z, photo in enumerate(new_x):
    photo *= 255.
    photo = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)
    points = np.array(out[z] * img_width, dtype=np.int32)
    cvt_point = []
    i = 0
    while i < len(points):
        cvt_point.append((points[i], points[i + 1]))
        i += 2
    cvt_point = np.array(cvt_point, dtype=np.int32)
    cv2.polylines(photo, [cvt_point], True, 255, 3)
    cv2.imwrite("ai_out3/"+str(z)+".jpg", photo)

# # pass
# # # list(map(lambda x: [x[0], x[1]], list(zip(x, y)))))
# # #
# # # for elem in train_ds:
# # #     print(elem.shape)
