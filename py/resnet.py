import numpy as np
import pickle
import cv2
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
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.applications.resnet_v2 import ResNet152V2 as imported_network
import tensorflow as tf

# https://stackoverflow.com/questions/70048701/how-can-i-preprocess-a-tf-data-dataset-using-a-provided-preprocess-input-functio
# Some sort of hack that takes a DS store and displays the image
def display(ds):
    images, _ = next(iter(ds.take(1)))
    image = images[0].numpy()
    image /= 255.0
    #plt.imshow(image)

# Some weird mapper, same source as above
def preprocess(images, labels):
    return preprocess_input(images), labels

EPOCHS = 2000
BS = 5
image_size = 0
directory_root = '../../local_data/symptoms on almond leaves/'
width=224
height=224
depth=3
default_image_size = tuple((width, height)) # Sets image size
LABELS='categorical'
SEED_LIST=(1,2,3)

for SEED in SEED_LIST:

    # Use Python scripting, but not Tensorflow, to determine  the number of classes by counting the number of sub
    # -directories
    n_classes = len(next(os.walk( directory_root ))[1])
    print( "Number of detected classes is " + str( n_classes ) )

    # New version: User image_dataset_from_directory
    train_ds = tf.keras.utils.image_dataset_from_directory(
            directory_root,
            batch_size=BS,
            image_size=(width,height),
            validation_split=0.2,
            crop_to_aspect_ratio=True,
            seed=6118,
            subset="training")

    val_ds = tf.keras.utils.image_dataset_from_directory(
            directory_root, 
            batch_size=BS,
            image_size=(width,height),
            validation_split=0.2,
            crop_to_aspect_ratio=True,
            seed=6118,
            subset="validation")

# per https://www.tensorflow.org/tutorials/load_data/images the following code uses a cache to speed up I/O on
# the disk
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Preprocess inputs
# https://stackoverflow.com/questions/70048701/how-can-i-preprocess-a-tf-data-dataset-using-a-provided-preprocess-input-functio
    train_ds = train_ds.map(preprocess)
    val_ds = val_ds.map(preprocess)

# Normalization of image intensities from [0,1] is perfomed by adding a special layer

    base_model = imported_network(
        weights='imagenet',                 # Load weights pre-trained on ImageNet.
        input_shape=(height, width, depth), # VGG16 expects min 32 x 32
        include_top=False)                  # Do not include the ImageNet classifier at the top.
    base_model.trainable = False

# Transfer learning example

    model = tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip(mode="horizontal"),                                          # Augmentation
        tf.keras.layers.RandomRotation(factor=(-0.25,0.25)),                                    # Augmentation
        tf.keras.layers.RandomTranslation(height_factor=(-0.1,0.1),width_factor=(-0.1,0.1)),     # Augmentation
        base_model,
        Flatten(),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(n_classes)
    ])

    opt = tf.keras.optimizers.Adam()
# distribution
    model.compile(
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=opt,
            metrics=[tf.metrics.SparseCategoricalAccuracy()]
            )
# train the network
    print("[INFO] training network...")

    model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            verbose=2)
