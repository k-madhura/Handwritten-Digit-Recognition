from keras.datasets import mnist
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Loading MNIST dataset
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Displaying the dataset shape and classes
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

if dr1==2:

    ## PCA
    print("\nPCA in progress >>> ")
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

## Implementing the SVM
K=['linear', 'rbf', 'polynomial']
dr2=0 # For selecting kernel 0-linear, 1-rbf, 2-polynomial
if K[dr2]=='polynomial':
    svclassifier =SVC(kernel='poly', degree=2)
elif K[dr2]=='rbf':
    svclassifier =SVC(kernel='rbf')
else:
    svclassifier =SVC(kernel='linear')  
    
svclassifier.fit(data_train, train_y)
print('For', K[dr2], 'classifier')
## Testing
y_pred =svclassifier.predict(data_test)
print(confusion_matrix(test_y, y_pred))
print(classification_report(test_y, y_pred))


