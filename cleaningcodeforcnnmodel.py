import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# Importing Deep Learning Libraries

from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Input,Dropout,GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Activation,MaxPooling2D
from keras.models import Model,Sequentiala
from keras.optimizers import Adam,SGD,RMSprop

from google.colab import drive
drive_path = '/content/gdrive/'
drive.mount(drive_path)
folder_path = os.path.join(drive_path, "MyDrive", "emotional detection", "data set")

picture_size = 48
batch_size = 120
datagen_train  = ImageDataGenerator(rotation_range=20,
                                    #brightness_range=None #(0 , 255)
                                    )

train_set = datagen_train.flow_from_directory(folder_path+"/test/bored-tired",
                                              target_size = (picture_size,picture_size),
                                              color_mode = "grayscale",
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              keep_aspect_ratio=True,
                                              save_to_dir=folder_path +"/testBored",
                                              shuffle=True)
len(train_set)

#Saving the generated cleaned data
counter = 0;
max_value = len(train_set)
for batch in train_set:
    counter+=1
    print(f'{counter}/{max_value}')
    if(counter == len(train_set)):
      break





