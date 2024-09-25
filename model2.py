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
for file in os.listdir("train/images"):
    im = cv2.imread("train/images/" + file)
    im = cv2.resize(im, (img_width, img_height))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(np.float32)
    im /= 255.
    x.append(im)

    with open("train/labels/" + file[:-4] + ".txt", "r") as f:
        # print(f.read().split(" "))
        y.append(np.array(list(map(float, f.read().split(" ")[1:])), dtype=np.float32))
x = np.array(x)
y = np.array(y)
for i in y:
    print(i.shape)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(img_width, img_height,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(8, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['accuracy'])

model.fit(x, y, epochs=100, batch_size=16)
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
    cv2.imwrite("ai_out2/"+str(z)+".jpg", photo)

# model.save_weights("output_model.ckpt")
model.save('./out_model')
# # pass
# # # list(map(lambda x: [x[0], x[1]], list(zip(x, y)))))
# # #
# # # for elem in train_ds:
# # #     print(elem.shape)
