import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Read from pickle file
train = pd.read_pickle('wheats_train.pickle')
test = pd.read_pickle('wheats_test.pickle')

# Split dataset as train and test
x_train = train.iloc[:,:-4]
y_train = train.iloc[:,-4:]

x_test = test.iloc[:,:-4]
y_test = test.iloc[:,-4:]
# define the model
model = Sequential()
model.add(Dense(10, input_dim=x_train.shape[1], activation='relu', kernel_initializer='he_normal'))
model.add(Dense(y_test.shape[1], activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=2)
# evaluate the keras model
_, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))

# Write predict value
y_predict = model.predict(x_test, batch_size=128)
print(y_test)
print(y_predict)

#A similiar matrix with zeros
pred = np.zeros_like(y_predict)
#max values fill with 1
pred[np.arange(len(y_predict)), y_predict.argmax(1)] = 1
#create classfication report
print(classification_report(y_test.values.argmax(axis=1),pred.argmax(axis=1)))

#confusion matrix 
wheats = ["AhmetAga","Bayraktar", "Bezostaya", "DropiTarex"]
conf_mat = confusion_matrix(y_test.values.argmax(axis=1),pred.argmax(axis=1))
print("Confusion Matrix:")
print(conf_mat)

ax= plt.subplot()
sns.heatmap(conf_mat, annot=True, ax = ax, fmt = 'g', cmap='Blues'); #annot=True to annotate cells
ax.xaxis.set_ticklabels(wheats);
ax.yaxis.set_ticklabels(wheats);
plt.savefig('confusion_matrix.png')
plt.show()



