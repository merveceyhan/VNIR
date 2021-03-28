import pandas as pd
import glob
import os
import numpy as np

# Read wheat files and and arrange values
def read_wheat(wheat_name):
    path = "BugdayOlcum_CSV/"+ wheat_name
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
        big_frame["Type"] = wheat_name
        # if(wheat_name == "AhmetAga"):
        #     big_frame["Type"] = 1
        # if(wheat_name == "Bayraktar"):
        #     big_frame["Type"] = 2
        # if(wheat_name == "Bezostaya"):
        #     big_frame["Type"] = 3
        # if(wheat_name == "DropiTarex"):
        #     big_frame["Type"] = 4 
            
    #big_frame.rename({0: 'Reflectance (AU)'}, axis=1, inplace=True)
    print("Test2: ")
    print(big_frame.head())
    return big_frame


# "AhmetAga","Bayraktar","Bezostaya" are used for wheat classification as a model
wheats = ["AhmetAga","Bayraktar", "Bezostaya", "DropiTarex"]

all_wheats = pd.DataFrame()

for i in range(0,len(wheats)):
   all_wheats=all_wheats.append(read_wheat(wheats[i]))

print(all_wheats.head())
# Save all wheats as a pickle file
all_wheats.to_pickle('export_allwheats.pickle')






    
