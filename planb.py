import os.path
import pickle

import PIL
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils import img_to_array, load_img
import pandas as pd

with open("dictionary/id2label_final.pkl", "rb") as file:
    labels_names = pickle.load(file)

img_height = 1000
img_width = 770
batch_size = 32
validation_split = 0.2

datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=validation_split
)

train_gen = datagen.flow_from_directory(
    'train_set/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_gen = datagen.flow_from_directory(
    'train_set/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

if os.path.isfile('model.h5'):
    model = tf.keras.models.load_model('model.h5')

    model.summary()

else:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(21, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'model-{epoch:03d}.h5',
    monitor='val_loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode='auto'
)

last = tf.keras.callbacks.ModelCheckpoint(
    'model.h5',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto'
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True
)

epochs = 100
history = model.fit(
    train_gen,
    validation_data=validation_gen,
    epochs=epochs,
    callbacks=[checkpoint, last, early_stopping]
)


def load_type_id(file_path: str) -> int:
    id_type = None
    try:
        id_type = int(labels_names[file_path])
    except KeyError:
        pass
    return id_type


model.save('model.h5')

model = tf.keras.models.load_model('model.h5')

classes = train_gen.class_indices
classes = {value: key for key, value in classes.items()}

results = pd.DataFrame(columns=['filename', 'predicted_class'])

for filename in os.listdir('test_set/'):
    path = os.path.join('test_set/', filename)
    try:
        img = load_img(path, target_size=(img_height, img_width))
        img_array = img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)

        img_preprocessed = img_batch / 255.
        prediction = model.predict(img_preprocessed)
        predicted_class = classes[np.argmax(prediction)]

        type_id = load_type_id(predicted_class)

        results.loc[len(results)] = [filename, type_id]
    except PIL.UnidentifiedImageError as e:
        print(e)

results.to_csv('predictions.csv', index=False)
