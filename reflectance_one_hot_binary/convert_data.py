import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Read wheat files and and arrange values
def read_wheat(wheat_name):
    path = "../BugdayOlcum_CSV/"+ wheat_name
    allFiles = glob.glob(os.path.join(path,"*.csv"))
    col_names = ['Reflectance (AU)']

    np_array_list = []
    

    for file in allFiles:
        df = pd.read_csv(file,names=col_names)
        df = df.iloc[1:]
        np_list = []
        np_list2 = []
        sum = float(0)
        for i in range(0,228):
            np_list.append(float(df.values[i][0]))
            sum = sum + float(df.values[i][0])
        if(sum>0):
            np_list2.append(np_list)
            np_array_list.append(np.array(np_list))

    big_frame = pd.DataFrame(np.array(np_array_list))
    for index in range(0,len(allFiles)):
        big_frame[wheat_name] = 1

    return big_frame


# "AhmetAga","Bayraktar","Bezostaya" are used for wheat classification as a model
wheats = ["AhmetAga","Bayraktar", "Bezostaya", "DropiTarex"]

train_all_wheats = pd.DataFrame()
test_all_wheats = pd.DataFrame()

for i in range(0,len(wheats)):
    df = read_wheat(wheats[i])
    train, test = train_test_split(df, test_size=0.10, random_state = 1)
    train_all_wheats=train_all_wheats.append(train)
    test_all_wheats=test_all_wheats.append(test)
    print(wheats[i], " completed")

for i in range(0,len(wheats)):
    train_all_wheats[wheats[i]] = train_all_wheats[wheats[i]].fillna(0)
    test_all_wheats[wheats[i]] = test_all_wheats[wheats[i]].fillna(0)

train_all_wheats = shuffle(train_all_wheats)
print(train_all_wheats.head())
test_all_wheats = shuffle(test_all_wheats)
print(test_all_wheats.head())

# Save all wheats as a pickle file
train_all_wheats.to_pickle('wheats_train.pickle')
test_all_wheats.to_pickle('wheats_test.pickle')

train_all_wheats.to_csv('wheats_train.csv')
test_all_wheats.to_csv('wheats_test.csv')






    
