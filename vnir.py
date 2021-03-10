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
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

### READ DATA


# "AhmetAga","Bayraktar","Bezostaya" are used for wheat classification as a model
wheats=["AhmetAga","Bayraktar"]

np_array_list = []

# read wheat files and and arrange values
def read_wheat(wheat_name):
    path = "BugdayOlcum_CSV\\"+ wheat_name
    allFiles = glob.glob(os.path.join(path,"*.csv"))
    col_names = ["Wavelength (nm)", "Absorbance (AU)", "Reference Signal (unitless)" , "Sample Signal (unitless)"]
        
    for file in allFiles:
        df = pd.read_csv(file,names=col_names)
        df = df.drop([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22])
        np_array_list.append(df.values)
                        
    #comb_np_array = np.vstack(np_array_list)
    big_frame = pd.DataFrame(np_array_list)
    
    
    for index in range(0,len(allFiles)):
        if(wheat_name == "AhmetAga"):
            big_frame["Type"]=1
        if(wheat_name == "Bayraktar"):
            big_frame["Type"]=2
        if(wheat_name == "Bezostaya"):
            big_frame["Type"]=3    
    big_frame.columns = ["Spectrum", "Type"]
    return big_frame


all_wheats=pd.DataFrame()

for i in range(0,len(wheats)):
    all_wheats=all_wheats.append(read_wheat(wheats[i]))
                       
# save all wheats as csv file
all_wheats.to_pickle('BugdayOlcum_CSV\export_allwheats.pickle')





# read from csv file
all_wheats = pd.read_pickle('BugdayOlcum_CSV\export_allwheats.pickle')
 
print(all_wheats.head())
print(all_wheats['Type'].value_counts())


# split dataset as train and test
train, test = train_test_split(all_wheats, test_size=0.10, random_state = 5)

#createx_train, x_test, y_train, y_test
x_train= train["Spectrum"]
x_test= test["Spectrum"]
y_train=train['Type'].astype(np.int32)
y_test=test['Type'].astype(np.int32)


# define sequential model
def get_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(4,229)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


yhat = get_model().predict(x_test)
for i in range(0, len(yhat)):
    print(yhat[i])


num_epochs =1
batch_size = 128
history = get_model().fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    validation_split=0.1)





"""		
yhat = yhat.round()

acc = accuracy_score(y_test, yhat)
		
print('>%.3f' % acc)
"""
