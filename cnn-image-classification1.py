

# wheat - image classification
# https://github.com/sirainatou/Image-classification-using-CNN-Vgg16-keras 

import numpy as np
import pandas as pd
import cv2
import os
from matplotlib import pyplot as plt
import pathlib
import tensorflow as tf
import glob

import math
from keras import applications
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Convolution2D,Activation,Flatten,Dense,Dropout,MaxPool2D,BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator 


data = []
labels = []

for folder in ['Tosunbey','Krasunia','Kirac66']:
    images = glob.glob('BugdayImg_VGG/%s/*.*'%folder)

    for i in range(len(images)): 
        image = cv2.imread(images[i])
        try : 
            image = cv2.resize(image, (100,100))
            image = tf.keras.preprocessing.image.img_to_array(image)
            data.append(image)
            labels.append(folder)
        except : 
            pass
    print(len(images),folder)
    


data = np.array(data)
label = np.array(labels)
print('\ndata.shape ')
print(data.shape)



dict_ = {'Tosunbey':0,
       'Krasunia':1,
       'Kirac66':2}
label_dict = {0: 'Tosunbey', 1: 'Krasunia',2: 'Kirac66'}
label = np.vectorize(dict_.get)(label)
print('\nlabel.shape ')
print(label.shape)



train_images_reshaped = data.reshape((-1, 100, 100,3))
train_labels_reshaped = to_categorical(label, num_classes=3)

# Convolution Neural Networks (CNN)
model = Sequential()

model.add(Conv2D(8, kernel_size=(3,3), padding='same', input_shape = (100,100,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(3, 3)))

model.add(Conv2D(16, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


model.fit(train_images_reshaped, train_labels_reshaped, epochs=20)

model.save_weights('vgg_cnn_model1.h5')

batch_size = 16

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(
        'BugdayImg_VGG/Train_data',  
        target_size=(100, 100),  
        batch_size=batch_size,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=20)
model.save_weights('vgg_cnn_data_augmentation.h5')  # always save your weights after training or during training



batch_size = 1
test_generator = test_datagen.flow_from_directory(
    'BugdayImg_VGG/Test_data',
    color_mode = "rgb",
    target_size=(100, 100),
    batch_size=1, 
    shuffle=True)
y_pred = model.predict_generator(test_generator, 1//batch_size, workers=3)


x_test, y_test = next(test_generator)
predicted = model.predict(x_test)
expected = y_test

print(predicted )
predicted  = predicted .argmax()
plt.imshow(x_test[0])
plt.title(label_dict[predicted])



