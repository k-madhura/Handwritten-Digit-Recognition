import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from IPython.display import Image, display
from keras.preprocessing.image import ImageDataGenerator
from os import listdir
import matplotlib
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from keras.applications import vgg16
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model, Sequential
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Activation, BatchNormalization, GlobalAvgPool2D, MaxPooling2D, Dropout

os.chdir(r'C:\Users\madhu\Desktop\New folder\633 project 2\input')  # Enter input directory here
print(os.listdir(r"C:\Users\madhu\Desktop\New folder\633 project 2\input"))     # Enter input directory here
raw = pd.read_csv(r'C:\Users\madhu\Desktop\New folder\633 project 2\input\monkey_labels.txt', skipinitialspace=True)    # Enter label directory here

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

# Pre processing data
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
                                                    shuffle=True, 
                                                    class_mode="categorical")
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), 
                                                    batch_size = BATCH_SIZE, 
                                                    shuffle=True,
                                                    class_mode="categorical")
train_num = train_generator.samples
validation_num = validation_generator.samples
num_classes = 10

# Building model
def get_net(num_classes):

    model = Sequential()
    model.add(Dense(3, activation='relu', input_shape=(150, 150, 3)))
    model.add(Dense(5, activation='relu'))  
    model.add(Conv2D(num_classes, (1, 1)))
    model.add(GlobalAvgPool2D())
    model.add(Activation('softmax'))
    
    return model

epochs = 20
batch_size=64
num_classes = 10
net = get_net(num_classes)
net.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

#Evaluating model
history = net.fit(train_generator,
                              steps_per_epoch= train_num // batch_size,
                              epochs=epochs,
                              validation_data=train_generator,
                              validation_steps= validation_num // batch_size,
                              verbose = 1
                             )

#Displaying metrics
def visualized_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.title('Training and validation accuracy')
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, val_acc, 'blue', label='Validation acc')
    plt.legend()

    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.plot(epochs, val_loss, 'blue', label='Validation loss')
    plt.legend()
    
    plt.show()
    
visualized_history(history)


