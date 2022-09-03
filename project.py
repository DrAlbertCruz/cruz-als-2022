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
from tensorflow.keras.models import Sequential                          # OK in 2.3
from tensorflow.keras.layers import BatchNormalization                  # OK in 2.3
from tensorflow.keras.layers import Conv2D                              # In 2.3 there is no .convolutional
from tensorflow.keras.layers import MaxPool2D                           # ^
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt                                       # Not needed

EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256)) # Sets image size
image_size = 0
directory_root = 'PlantVillage-Dataset/raw/color'
width=256
height=256
depth=3

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, default_image_size) # Resize the image to a square image
            # TODO: Squareify the image but maintain its aspect ratio?
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print( "Error : {e}" )
        return None

# Initialize the structures to hold the image data and the labels
image_list, label_list = [], []

try:
    print( "Running: Loading images ..." )
    print( " ... directory to load is: ", directory_root )
    root_dir = listdir( directory_root )
    print( " ... it has ", str( len( root_dir ) ), " sub-directories." )
    for directory in root_dir: # Directory is treated as the "class name"
        print( " ... ... Currently loading class: ", directory )
        class_root = os.path.join( directory_root, directory )
        print( class_root )
        class_image_list = listdir( class_root )
        print( " ... ... It has ", str( len( class_image_list ) ), " images." )
        for image in class_image_list:
            image_file = os.path.join( class_root, image )
            if (
                    image_file.endswith(".jpg") == True or 
                    image_file.endswith(".JPG") == True or 
                    image_file.endswith(".png") == True or
                    image_file.endswith(".PNG")
                ):
                image_list.append( convert_image_to_array(image_file) )
                label_list.append( directory )
                print(" ... ... ... Image loading completed: ", image_file )
except Exception as e:
    print("Error : ", e)

exit()

image_size = len(image_list)

label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)

#print(label_binarizer.classes_)

np_image_list = np.array(image_list, dtype=np.float16) / 225.0

print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")

model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))

model.summary()
#print(model.summary)
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["acc"])
# train the network
print("[INFO] training network...")

history = model.fit(aug.flow(x_train, y_train, batch_size=BS), validation_data=(x_test, y_test), steps_per_epoch=len(x_train) // BS, epochs=EPOCHS, verbose=1)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
print("Starting plots")
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print("Test Accuracy: {scores[1]*100}")

# save the model to disk
print("[INFO] Saving model...")
pickle.dump(model,open('result_test.txt', 'wb'))
print("Project complete steph")
