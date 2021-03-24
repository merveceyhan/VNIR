import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import glob
import os

from numpy import mean
from numpy import std

from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold,train_test_split

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Embedding,MaxPooling1D,Flatten,Input
from keras.layers.convolutional import Convolution1D,AveragePooling1D,MaxPooling1D 
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D 




# "AhmetAga","Bayraktar","Bezostaya" are used for wheat classification as a model
wheats = ["AhmetAga","Bayraktar","Bezostaya","DropiTarex"]


# Read wheat files and and arrange values
def read_wheat(wheat_name):
    path = "BugdayOlcum_CSV\\"+ wheat_name
    allFiles = glob.glob(os.path.join(path,"*.csv"))
    col_names = ['Reflectance (AU)']
     
    np_array_list = []
        
    for file in allFiles:
        df = pd.read_csv(file,names=col_names)
        df = df.iloc[1:]
        np_array_list.append(df.values)

    big_frame = pd.DataFrame()
    big_frame = pd.DataFrame(np_array_list)
    
           
    for index in range(0,len(allFiles)):
        if(wheat_name == "AhmetAga"):
            big_frame["Type"] = 1
        if(wheat_name == "Bayraktar"):
            big_frame["Type"] = 2
        if(wheat_name == "Bezostaya"):
            big_frame["Type"] = 3
        if(wheat_name == "DropiTarex"):
            big_frame["Type"] = 4 
            
    big_frame.rename({0: 'Reflectance (AU)'}, axis=1, inplace=True)
   
    return big_frame


all_wheats = pd.DataFrame(columns=['Reflectance (AU)', 'Type'])

for i in range(0,len(wheats)):
    all_wheats=all_wheats.append(read_wheat(wheats[i]))
                     
# Save all wheats as csv file
all_wheats.to_csv('BugdayOlcum_CSV\export_allwheats.csv', index=False)

# Read from csv file
all_wheats = pd.read_csv('BugdayOlcum_CSV\export_allwheats.csv', encoding= 'unicode_escape',delimiter = ',')

# Split dataset as train and test
train, test = train_test_split(all_wheats, test_size=0.10, random_state = 5)

# Create x_train, x_test, y_train, y_test and convert "\\n" character to comma
x_train = pd.DataFrame(train["Reflectance (AU)"])
x_train = x_train.iloc[1:]
x_train = x_train.replace({'\\n ': ','}, regex=True)

x_test= pd.DataFrame(test["Reflectance (AU)"])
x_test = x_test.iloc[1:]
x_test = x_test.replace({'\\n ': ','}, regex=True)

y_train=pd.DataFrame(train['Type']).astype(np.int32)
y_test=pd.DataFrame(test['Type']).astype(np.int32)


clean_list = []
data_list = []
data_vector = []

# Clean values for special characters and split 
def Convert(string): 
    li = list(string.split("\\n"))
    for x in li:
        x = x.replace("['", "")
        x = x.replace("']", "")
        x = x.replace("[", "")
        x = x.replace("]", "")
        x = x.replace("'", "") 
        x = x.split(",")
        clean_list.append(x)
            
    return clean_list 



def Convert_AllData(data):
    # Get first 1000 values from x_train for trial
    # Then the value of 100 will be updated as "len(data)"
    for i in range(0, 100):
        converted_data_list = Convert(data.iat[i,0])
        for j in range(0, len(converted_data_list)):
            data_list.append(pd.to_numeric(converted_data_list[j]))
        
    for k in range(0,len(data_list)):
        data_vector.append(list(data_list[k]))
        
    return data_vector

# Convert x_train to dataframe for get matrix
x_train = pd.DataFrame(Convert_AllData(x_train))

# Convert x_test to dataframe for get matrix
x_test =pd.DataFrame(Convert_AllData(x_test))


# Define sequential model
def get_model():
    model = Sequential()
    activation = 'relu'
    model.add(Embedding(input_dim=2, output_dim=1, input_length=228))
    model.add(Convolution1D(128, 5, input_shape=(228,1), activation=activation))

    
    model.add(Convolution1D(128, 5, activation=activation))

    
    model.add(Convolution1D(64, 3, activation=activation))

    
    model.add(Convolution1D(64, 3, activation=activation))

    
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss = "binary_crossentropy", optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())
    print("CNN Model created.")
    
    return model


model = get_model()

# Params 
epochs = 5
batch_size = 128
seed = 7

# Fit and run our model
np.random.seed(seed)
hist = model.fit(x_train.iloc[0:5000,:],y_train.iloc[0:5000,:],validation_data=(x_test.iloc[0:1200,:], y_test.iloc[0:1200,:]),epochs=epochs,batch_size=batch_size,shuffle = True,verbose=2)





    