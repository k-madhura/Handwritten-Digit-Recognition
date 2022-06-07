import numpy as np 
import pandas as pd 
import os
from IPython.display import Image, display
from keras.preprocessing.image import ImageDataGenerator
from os import listdir
import matplotlib
from matplotlib import pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model, Sequential
from keras import optimizers
from keras.applications import vgg16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Conv2D, Activation, BatchNormalization, GlobalAvgPool2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping

os.chdir(r'C:\Users\madhu\Desktop\New folder\633 project 2\input') # Enter input directory here
print(os.listdir(r"C:\Users\madhu\Desktop\New folder\633 project 2\input")) # Enter input directory here
raw = pd.read_csv(r'C:\Users\madhu\Desktop\New folder\633 project 2\input\monkey_labels.txt', skipinitialspace=True) # Enter label directory here
raw = raw.rename(columns=lambda x: x.strip())
labels = pd.DataFrame()
labels["id"] = raw["Label"].str.strip()
labels["name"] = raw["Common Name"].str.strip()

TRAIN_DIR = "../input/training/training/"
VALIDATION_DIR = "../input/validation/validation/"
print(os.listdir(TRAIN_DIR))

all_ids = labels["id"]

for my_id in labels["id"]:
    images_to_show = 5
    image_dir = TRAIN_DIR + "%s/" % my_id
    image_name = listdir(image_dir)[0]
    image_path = image_dir  + image_name
    print("Labels is %s" % my_id)
    display(Image(filename=image_path, width=300, height=300))

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
BATCH_SIZE = 16

## Preprocessing dataset
train_datagen = ImageDataGenerator(rescale=1./255,      # Normalizing
                                    rotation_range=40,      
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True, 
                                    fill_mode='nearest'
                                  )

validation_datagen = ImageDataGenerator(rescale=1./255, # Normalizing
                                  )

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), 
                                                    batch_size = BATCH_SIZE, 
                                                    shuffle=True, # By shuffling the images we add some randomness and prevent overfitting
                                                    class_mode="categorical")
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), 
                                                    batch_size = BATCH_SIZE, 
                                                    shuffle=True,
                                                    class_mode="categorical")
train_num = train_generator.samples
validation_num = validation_generator.samples
num_classes = 10
# Importing VGG16
vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                     input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
output = vgg.layers[-1].output
output =keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)
vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False

# Building model    
model = Sequential()
model.add(vgg_model)
model.add(Dense(10, activation='relu'))

print(model.summary())
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Early stopping criteria
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)
batch_size=16
epochs=10
train_1=model.fit(train_generator, steps_per_epoch= train_num // batch_size, epochs=epochs,
                              validation_data=train_generator,
                              validation_steps= validation_num // batch_size,
                              verbose = 1, callbacks=[es])
# Evaluating model
scores = model.evaluate(validation_generator)
print("Transfer learning Error: %.2f%%" % (100-scores[1]*100))
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

## Displaying metrics
accuracy = train_1.history['accuracy']
val_accuracy = train_1.history['val_accuracy']
loss = train_1.history['loss']
val_loss = train_1.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'red', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'blue', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

