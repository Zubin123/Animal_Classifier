import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
#from tensorflow.keras.preprocessing import image

# Load the trained model
MODEL_PATH = "models/cnn_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define dataset path
DATASET_PATH = "dataset/"  # Update this if needed
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load test dataset (if available)
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Normalize test dataset
normalization_layer = tf.keras.layers.Rescaling(1./255)
class_names = test_dataset.class_names
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Function to predict a single image
def predict_image(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

# Test on a sample image
SAMPLE_IMAGE_PATH = "pexels-yaroslav-shuraev-8499870.jpg"  # Change this to your test image path
if os.path.exists(SAMPLE_IMAGE_PATH):
    #class_names = test_dataset.class_names
    predicted_class, confidence = predict_image(SAMPLE_IMAGE_PATH, model)
    print(f"Predicted Class: {class_names[predicted_class]} with {confidence:.2f} confidence")
else:
    print("Sample image not found. Place an image at the given path for testing.")
