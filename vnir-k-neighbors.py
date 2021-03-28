import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense,Flatten, Embedding,Conv2D,MaxPooling2D,Dropout
from keras.layers.convolutional import Convolution1D,AveragePooling1D
from keras.layers.normalization import BatchNormalization


# Read from pickle file
all_wheats = pd.read_pickle('export_allwheats.pickle')
print(all_wheats.head())

# Split dataset as train and test
train, test = train_test_split(all_wheats, test_size=0.10, random_state = 10)


# Create x_train, x_test, y_train, y_test
x_train = pd.DataFrame(train.iloc[:,0:228])
x_train = x_train.astype("float32")
x_test = pd.DataFrame(test.iloc[:,0:228])
x_test = x_test.astype("float32")

y_train = pd.DataFrame(train['Type'])
y_test = pd.DataFrame(test['Type'])

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

clf = KNeighborsClassifier(1,weights='distance') 
clf.fit(x_train,np.ravel(y_train)) 
y_predict = clf.predict(x_test)
print(y_test.transpose().values)
print(y_predict)

import seaborn as sns

wheats = ["AhmetAga","Bayraktar", "Bezostaya"]
conf_mat = confusion_matrix(y_test,y_predict,labels=wheats)

#conf_mat = conf_mat.T # since rows and  cols are interchanged
avg_acc = np.trace(conf_mat)/len(y_test)
print("Accuracy: ", avg_acc)
print("Confusion Matrix:")
print(conf_mat)

# Viewing the confusion matrix
import matplotlib.pyplot as plt
ax= plt.subplot()
sns.heatmap(conf_mat, annot=True, ax = ax, fmt = 'g', cmap='Blues'); #annot=True to annotate cells
ax.xaxis.set_ticklabels(wheats);
ax.yaxis.set_ticklabels(wheats);
plt.savefig('confusion_matrix.png')
plt.show()


# model = Sequential()
# activation = 'relu'
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=((1,228,1))))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(3, activation='softmax'))
# model.compile(loss = "binary_crossentropy", optimizer='adam', metrics=['accuracy'])

# print(model.summary())
# print("CNN Model created.")

# # Params 
# epochs = 10
# batch_size = 32
# seed = 7

# # Fit and run our model
# np.random.seed(seed)
# hist = model.fit(x_train,y_train,validation_split=0.1, epochs=epochs,batch_size=batch_size,shuffle = True,verbose=2)

# results = model.evaluate(x_test, y_test, batch_size=128)
# predicted = model.predict(x_test)

# print(results)
# print(predicted)



   

   
