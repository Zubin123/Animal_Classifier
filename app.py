import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, url_for
from PIL import Image

# Load model
MODEL_PATH = "models/cnn_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names
CLASS_NAMES = ["Bear", "Bird", "Cat", "Cow", "Deer", "Dog", "Dolphin", "Elephant", "Giraffe", "Horse", "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"]

# Upload folder inside static for serving images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)  # No need to set template folder manually
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        
        # Save uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Process and predict
        img = preprocess_image(file_path)
        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        confidence = np.max(predictions)
        
        result = f"{CLASS_NAMES[class_index]} ({confidence:.2f} confidence)"
        return render_template('index.html', result=result, image_url=url_for('static', filename=f'uploads/{file.filename}'))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
