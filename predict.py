import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model('model.h5')

# Take input from user
img_path = input("Enter image path: ")

# Load image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Prediction
prediction = model.predict(img_array)[0][0]

# Result
if prediction > 0.5:
    print(f"Result: DEFECT ❌")
    print(f"Confidence: {round(prediction*100, 2)}%")
else:
    print(f"Result: PASS ✅")
    print(f"Confidence: {round((1-prediction)*100, 2)}%")