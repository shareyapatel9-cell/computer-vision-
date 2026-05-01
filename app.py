from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load model
model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    # Load and preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        result = "DEFECT ❌"
        confidence = round(prediction * 100, 2)
    else:
        result = "PASS ✅"
        confidence = round((1 - prediction) * 100, 2)

    return render_template('index.html', result=result, confidence=confidence, img_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)