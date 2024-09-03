import os
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as keras
import numpy as np
import cv2 as cv2
from tensorflow import keras
from keras import layers, models

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


dataset_dir = os.path.join(os.getcwd(), 'photos')

dataset_train_dir = os.path.join(dataset_dir, 'train')
dataset_train_correct_len = len(os.listdir(os.path.join(dataset_train_dir, 'correctphotos')))
dataset_train_wrong_len = len(os.listdir(os.path.join(dataset_train_dir, 'wrongphotos')))

dataset_validation_dir = os.path.join(dataset_dir, 'validation')
dataset_validation_correct_len = len(os.listdir(os.path.join(dataset_validation_dir, 'correct')))
dataset_validation_wrong_len = len(os.listdir(os.path.join(dataset_validation_dir, 'wrong')))


print('Train correct: %s' % dataset_train_correct_len)
print('Train wrong: %s' % dataset_train_wrong_len)
print('validation correct: %s' % dataset_validation_correct_len)
print('validation wrong: %s' % dataset_validation_wrong_len)

image_width = 3000
image_height = 4000
image_color_channel_size = 3
image_size = (image_width, image_height)
image_shape = image_size + (image_color_channel_size,)

batch_size = 32
epochs = 20
learnging_rate = 0.0001

class_names = ['correct', 'wrong']

dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_train_dir,
    image_size = image_size,
    batch_size = batch_size,
    shuffle = True
)

dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_train_dir,
    image_size = image_size,
    batch_size = batch_size,
    shuffle = True
)

dataset_validation_cardinality = tf.data.experimental.cardinality(dataset_validation)
dataset_validation_batches = dataset_validation_cardinality // 5

dataset_test = dataset_validation.take(dataset_validation_batches)
dataset_validation = dataset_validation.skip(dataset_validation_batches)

print('validation dataset cardinality: %d' % tf.data.experimental.cardinality(dataset_validation))
print('test dataset cardinality: %d' % tf.data.experimental.cardinality(dataset_test))
            
            
model = tf.keras.models.Sequential([
    tf.keras.experimental.preprocessing.Rescaling(
        1. / image_color_channel_size,
        input_shape = image_shape
    ),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.laeyers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(
    optimizer= tf.keras.optimizer.Adam(learnging_rate = learnging_rate),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics = ['accuracy']
              )

