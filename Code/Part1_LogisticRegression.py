from keras.datasets import mnist
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# load data
(train_x, train_y), (test_x, test_y) = mnist.load_data()

print('X_train: ' + str(train_x.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_x.shape))
print('Y_test:  '  + str(test_y.shape))

classes = np.unique(train_y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

## Resizing the training and testing matrix
num_pixels = train_x.shape[1] * train_x.shape[2]
train_x = train_x.reshape((train_x.shape[0], num_pixels)).astype('float32')
test_x = test_x.reshape((test_x.shape[0], num_pixels)).astype('float32')
scaler = StandardScaler()

# Fit on training set only and applying transform to both the training set and the test set.
data_train = scaler.fit_transform(train_x)
data_test = scaler.transform(test_x)
dr1=2# dr1=1 for PCA , dr1=2 for MDA

if dr1==1:
    ## PCA
    print("\nPCA in progress >>> ")
    # Make an instance of the Model
    pca = PCA(.95)
    pca.fit(data_train)
    data_train = pca.transform(data_train)
    data_test = pca.transform(data_test)
    print(">>> Done PCA\n")

else:
    ## MDA
    print("\nMDA in progress >>> ")
    lda = LDA()
    lda.fit(data_train, train_y)
    data_train = lda.transform(data_train)
    data_test = lda.transform(data_test)
    print(">>> Done MDA\n")

# apply logistic regressor 
clf = LogisticRegression(C=1e5,
                         multi_class='multinomial',
                         penalty='l2', solver='sag', tol=0.1)
# fit data
clf.fit(data_train, train_y)
y_pred=clf.predict(data_test)
print(confusion_matrix(test_y, y_pred))
print(classification_report(test_y, y_pred))

