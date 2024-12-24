from tensorflow.keras.models import Sequential
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Input
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tensorflow import keras
import tensorflow as tf


load_dotenv()
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
CHANNELS = int(os.getenv("CHANNELS"))
EPOCHS = int(os.getenv("EPOCHS"))

dataset_path = 'Dataset'
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
class_name= dataset.class_names

shuffle_size= min(2000, len(dataset))
def get_dataset_partition_tf(ds, train_split=0.7, val_split=0.2, test_split=0.1, shuffle=True, shuffle_size=shuffle_size):
    ds_size = len(ds)  
    if shuffle:
        ds = ds.shuffle(shuffle_size, reshuffle_each_iteration=False)  # Shuffle the dataset

    # Define the split sizes
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    
    # Split the dataset using take and skip
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size + val_size)

    return train_ds, val_ds, test_ds
train_ds, val_ds, test_ds= get_dataset_partition_tf(dataset)
#for image, label in train_ds.take(1):
#    print(f"Image batch shape: {image.shape}, Label batch shape: {label.shape}")
train_ds= train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds= val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds= test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE) 

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0/255)

])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])