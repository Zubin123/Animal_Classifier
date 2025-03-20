import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# ======= Load Dataset =======
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
DATASET_PATH = "dataset/"

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE)

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# ======= Define Model =======
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze pre-trained layers

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(15, activation='softmax')
])

# ======= Phase 1: Train Frozen Model =======
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history_1 = model.fit(train_dataset, validation_data=val_dataset, epochs=5)

# ======= Phase 2: Fine-Tune Model =======
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history_2 = model.fit(train_dataset, validation_data=val_dataset, epochs=5)

# ======= Save Model =======
os.makedirs("models", exist_ok=True)
model.save("models/cnn_model.keras")

# ======= Plot Accuracy =======
train_acc = history_1.history['accuracy'] + history_2.history['accuracy']
val_acc = history_1.history['val_accuracy'] + history_2.history['val_accuracy']

plt.figure(figsize=(10, 5))
plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend()
plt.title("Model Accuracy Over Epochs")
plt.savefig("training_accuracy_plot.png", dpi=300)
plt.close()
