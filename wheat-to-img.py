import os, random
import pandas as pd
import glob
import numpy as np
from PIL import Image as im
from matplotlib import pyplot as plt
import time


np_list = []
out = []

# READ DATA    
def read_wheat(wheat_name):
  
    # buğday verilerinin olduğu klasör yolu
    path = "BugdayOlcum_CSV/"+ wheat_name + "/"
    
    # klasörden random veri seçilir    
    file = random.choice(os.listdir(path))
    
    if os.stat(path + file).st_size == 0:
        print('empty')
        
    else:   
        col_names = ['Reflectance (AU)']   
        df = pd.read_csv(path + file, names=col_names)
        df = df.iloc[1:]
        
        np_list = []
        np_array_list = []
        sum = float(0)
        
        # verisetindeki 228 sütunluk verilerin ilk 224 sütunu okunuyor (sayıları denkleştirmek için)
        for i in range(0,224):
            np_list.append(float(df.values[i][0]))
            sum = sum + float(df.values[i][0])
        if(sum>0):
            np_array_list.append(np.array(np_list))
            np_array_list = np.concatenate(np_array_list).ravel().tolist()
            for i in np_array_list:
                out.append(i)
    
        # 224 * 224 lük resim için 50176 veri gerekir.
        if(len(out)<50176):
            read_wheat(wheat_name)
    
    
    return out

        
# CREATE IMAGE      
def create_img(out, wheat_name):
    # VGG modeli için 224*224 şeklinde yeniden şekillendirilir.
    out = np.reshape(out, (224, 224))
    
    # resimdeki değerler 255 e bölünür
    formatted = (out * 255 / np.max(out)).astype('uint8')
    data = im.fromarray(formatted)
    
    # oluşturulan her bir resim ayrı ayrı buğday klasörlerine kaydedilir.
    data.save("BugdayImg_VGG/" + wheat_name + "/" + str(time.strftime("%Y%m%d-%H%M%S")) + ".png", format="PNG")






wheats = ["AhmetAga","Bayraktar","Bezostaya","DropiTarex","Ekiz","Esperia","Flamura","Gerek79","Katea","Kirac66","Konya2002","Krasunia","Maden","Misiia","Mufitbey","Nacibey","Nota","Pehlivan","Quality","Rumeli","Sonmez","Syrena","Tosunbey","Yubileynaus"]

for wheat in wheats:
    # her sınıftan kaçar tane resim üretilmek istendiğidir 
    # bu döngüde her sınıftan 100 tane üretiyor
    for i in range(0,100):
        out = read_wheat(wheat)
        # 224*224 lük resim için 50176 veri içeren liste olusturulduktan sonra create_img() fonk. çağırılır.
        create_img(out, wheat)
        # oluşturulan her bir resimden sonra liste temzilenir.
        out.clear()
  
    

