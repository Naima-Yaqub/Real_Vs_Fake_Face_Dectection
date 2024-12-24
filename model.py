from Data_processing import resize_and_rescale, data_augmentation
from tensorflow.keras.models import Sequential
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import MobileNetV2
from Data_processing import class_name
from dotenv import load_dotenv
from tensorflow import keras
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

load_dotenv()
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
CHANNELS = int(os.getenv("CHANNELS"))
EPOCHS = int(os.getenv("EPOCHS"))


INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

# Building the model
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=INPUT_SHAPE),
    resize_and_rescale,
    #data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    #tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # L2 regularization
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
#model = models.Sequential([
#    Input(shape=INPUT_SHAPE),
#    resize_and_rescale,
#    data_augmentation,
#    layers.Conv2D(32, (3, 3), activation='relu'),
#    layers.MaxPooling2D((2, 2)),
#   layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#    layers.MaxPooling2D((2, 2)),
#    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#    layers.MaxPooling2D((2, 2)),
#    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#    layers.MaxPooling2D((2, 2)),
#    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#    layers.MaxPooling2D((2, 2)),
#    layers.Flatten(),
#    layers.Dense(256, activation='relu'),
#    tf.keras.layers.GlobalAveragePooling2D(),
#    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # L2 regularization
#    tf.keras.layers.Dropout(0.5),
#    tf.keras.layers.Dense(1, activation='sigmoid')
#    #layers.Dense(1600, activation='relu'),
#    #layers.Dense(64, activation='relu'),
#    #layers.Dense(len(class_name), activation='softmax')
#])

# Displaying the model summary
model.summary()


