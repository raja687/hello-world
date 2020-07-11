import keras
from keras.models import *
from keras.models import Sequential
import cv2
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)
model = Sequential() 
model.add(Conv2D(32, (2, 2), input_shape = input_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Conv2D(32, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Conv2D(64, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1)) 
model.add(Activation('sigmoid')) 
model.compile(loss ='binary_crossentropy',optimizer ='rmsprop',metrics =['accuracy']) 
model =load_model('model_saved.h5')
img = cv2.imread('18.jpg')
img = cv2.resize(img,(224,224))
img = np.reshape(img,[1,224,224,3])
classes = model.predict_classes(img)
a=classes
if(a[0]==0):
    print('it is car')
else:
    print('it is plane')

