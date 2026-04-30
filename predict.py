
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Load image
img_path = input("Enter image path: ")

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Prediction
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Result: DEFECT ❌")
else:
    print("Result: PASS ✅")