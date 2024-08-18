# Dataset from: https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset?resource=download

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
import pandas as pd

# Loading files/folders needed.
rawtrain = 'train'
rawtest = 'test'
rawvalid = 'valid'

train_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

training = train_data.flow_from_directory(
    rawtrain,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

testing = test_data.flow_from_directory(
    rawtest,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

valid = valid_data.flow_from_directory(
    rawvalid,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Create CNN
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
        optimizer = 'adam',
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )


model.fit(training, batch_size=32, epochs=10, validation_data=valid)
model.save('skin_types.keras')