!pip install tensorflow keras matplotlib numpy

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow_datasets as tfds

dataset_name = "cats_vs_dogs"
(train_ds, test_ds), dataset_info = tfds.load(dataset_name, split=['train[:80%]', 'train[80%:]'], as_supervised=True, with_info=True)

base_model = keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Congela os pesos do modelo pré-treinado

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')  # Saída binária: 0 ou 1 (Gato ou Cachorro)
])

# Resize images to a consistent size before batching
def resize_image(image, label):
    image = tf.image.resize(image, (160, 160))
    return image, label

train_ds = train_ds.map(resize_image)
test_ds = test_ds.map(resize_image)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds.batch(32), validation_data=test_ds.batch(32), epochs=5)

loss, accuracy = model.evaluate(test_ds.batch(32))
print(f"Acurácia: {accuracy:.2f}")

def predict_image(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img_array = keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)

    if prediction[0] > 0.5:
        print("É um Cachorro!")
    else:
        print("É um Gato!")

predict_image("/content/xxx.jpg")

model.save("transfer_learning_model.h5")