from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# loading data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshaping data
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
# normalize inputs
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#Building baseline modle
def baseline_model():
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compiling model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
model = baseline_model()
# Training the model
train_1= model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=8, batch_size=400)

# evaluation of the model
scores = model.evaluate(X_test, y_test)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Displaying metrics
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
