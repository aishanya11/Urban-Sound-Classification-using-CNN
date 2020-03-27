from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras import models
from keras import layers
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import numpy as np
from memory_profiler import memory_usage
import os
import pandas as pd
from glob import glob
import librosa
import librosa.display
import gc

import pyaudio
import wave

root_path = '/home/aishanya/Desktop/kaggle/'

image_height = 217
image_width = 223

model = models.Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(image_height,image_width,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.load_weights(root_path +'model1.hdf5')
validation_dir = root_path + 'working/test'
validation_datagen = ImageDataGenerator(rescale=1./255)
print(model.summary())
