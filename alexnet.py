import numpy as np      # ??? Uses numpy?
import pickle           # What is Pickle for?
# Pickle is used for serializing and deserializing Python objects. Most likely this is how we will save and restore
# ... models that have been trained
import cv2              # Guessing we use openCV to load images
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os import listdir
from sys import exit # Be able to quit

from tensorflow import keras

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential                          # OK in 2.3
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, Rescaling, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras import losses
from tensorflow.keras.utils import image_dataset_from_directory, img_to_array
# import matplotlib.pyplot as plt                                       # Not needed
import tensorflow as tf

EPOCHS = 100
BS = 5
image_size = 0
directory_root = '../../local_data/PlantVillage-Dataset/raw/color'
#directory_root = 'rice_leaf_diseases'
width=227
height=227
depth=3
default_image_size = tuple((width, height)) # Sets image size
LABELS='int'

# Use Python scripting, but not Tensorflow, to determine  the number of classes by counting the number of sub
# -directories
n_classes = len(next(os.walk( directory_root ))[1])

# New version: User image_dataset_from_directory
train_ds = tf.keras.utils.image_dataset_from_directory( 
        directory_root, 
        labels='inferred', 
        label_mode=LABELS, 
        batch_size=BS, 
        image_size=(width,height),
        validation_split=0.2,
        crop_to_aspect_ratio=True,
        seed=123,
        subset="training")

val_ds = tf.keras.utils.image_dataset_from_directory( 
        directory_root, 
        labels='inferred', 
        label_mode=LABELS, 
        batch_size=BS, 
        image_size=(width,height),
        validation_split=0.2,
        crop_to_aspect_ratio=True,
        seed=123,
        subset="validation")

# per https://www.tensorflow.org/tutorials/load_data/images the following code uses a cache to speed up I/O on
# the disk
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalization of image intensities from [0,1] is perfomed by adding a special layer

model = Sequential()
inputShape = (height, width, depth)
#chanDim = -1
#if K.image_data_format() == "channels_first":
#    inputShape = (depth, height, width)
#    chanDim = 1

# Intensity normalization layer
model.add( Rescaling(1./255) ) # Rescaling of image intensity handled here
# Layer 1
model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4,4), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3, 3), strides=(2,2)))
# Layer 2
model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3, 3), strides=(2,2)))
# Layer 3
model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1,1), activation="relu", padding="same"))
model.add(BatchNormalization())
# Layer 4
model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3, 3), strides=(2,2)))
# End of convolutional layers
model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(n_classes))

opt = Adam()
# distribution
model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), 
        optimizer=opt,
        metrics=tf.keras.metrics.SparseCategoricalAccuracy())
# train the network
print("[INFO] training network...")

model.fit( 
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS )

