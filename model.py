import cv2
import tensorflow as tf
import numpy as np

import sys

model = tf.keras.models.load_model("/Users/hongjowell/parking/Model/saved_model.h5")

from tensorflow.keras.applications.vgg16 import VGG16
HEIGHT = 49
WIDTH = 37

img = sys.argv[1]
img = "/Users/hongjowell/parking/" + img
print(sys.argv)
#img = "./parking/sample.JPG"

image = cv2.imread(img, cv2.IMREAD_COLOR)
image = cv2.resize(image, (WIDTH, HEIGHT))
image_x = np.expand_dims(image, axis=0)
image_x = tf.keras.applications.vgg16.preprocess_input(image_x)
prediction = model.predict(image_x)
prediction = np.squeeze(prediction)

print(prediction)

if prediction> 0.8:
    print('Occupied')
else:
    print('Empty')