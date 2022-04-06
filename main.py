import cv2 
import numpy as np
import tensorflow as tf

img = cv2.imread('/Users/hongjowell/parking/parkinglot.jpeg')

y = 149
x = 142
i = 0

model = tf.keras.models.load_model("/Users/hongjowell/parking/Model/saved_model.h5")

from tensorflow.keras.applications.vgg16 import VGG16
HEIGHT = 49
WIDTH = 37

for i in range (19):
    cropped_image = img [y:y+85, x:x+45]
    image = cv2.resize(cropped_image, (WIDTH, HEIGHT))
    image_x = np.expand_dims(image, axis=0)
    image_x = tf.keras.applications.vgg16.preprocess_input(image_x)
    prediction = model.predict(image_x)
    prediction = np.squeeze(prediction)

    if prediction> 0.8:
        cv2.rectangle(img, (x, y), (x+45, y+85),(0, 0, 255), 3)
        
    else:
        cv2.rectangle(img, (x, y), (x+45, y+85),(0, 255, 0), 3)

    x = x+45

cv2.imshow('parking lot', img)
cv2.waitKey(0)
cv2.destroyAllWindows()