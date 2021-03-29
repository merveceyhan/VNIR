import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Activation


# Read from pickle file
all_wheats = pd.read_pickle('export_allwheats_numbers.pickle')
print(all_wheats.head())

# Split dataset as train and test
train, test = train_test_split(all_wheats, test_size=0.10, random_state = 7)

# Create x_train, x_test, y_train, y_test
x_train = pd.DataFrame(train.iloc[:,0:228])
x_train = x_train.astype("float32")
x_test = pd.DataFrame(test.iloc[:,0:228])
x_test = x_test.astype("float32")

y_train = pd.DataFrame(train['Type'])
y_test = pd.DataFrame(test['Type'])


# Create Sequential Model
nSNP=x_train.shape[1] 
model = Sequential()
model.add(Dense(64, input_dim=nSNP))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('softplus'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.summary()
model.fit(x_train, y_train, epochs=50)

# Write predict value
y_predict = model.predict(x_test, batch_size=128)
print(y_predict)

# Draw Observed vs Predicted Y
mse_prediction = model.evaluate(x_test, y_test, batch_size=128)
print('\nMSE in prediction =',mse_prediction)
plt.title('MLP: Observed vs Predicted Y')
plt.ylabel('Predicted')
plt.xlabel('Observed')
plt.scatter(y_test, y_predict, marker='o')
plt.show()




