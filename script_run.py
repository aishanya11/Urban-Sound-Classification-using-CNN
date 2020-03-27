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
#%load_ext memory_profiler



# Getting the mapping from class index to class label
idx2label = {0: 'air_conditioner', 1: 'car_horn', 2: 'children_playing', 3: 'dog_bark', 4: 'drilling', 5: 'engine_idling', 6: 'gun_shot', 7: 'jackhammer', 8: 'siren', 9: 'street_music'}

def create_spectrogram(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = root_path + 'working/images/' + name + '.jpg'
    filename2 = '/home/aishanya/Desktop/FRONT END/views/' + name +'.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.savefig(filename2, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(image_height, image_width))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor




Data_dir=np.array(glob(root_path+'/working/test_sounds/*'))


for file in Data_dir:
    #print(file)
    filename,name = file,file.split('/')[-1].split('.')[0]
    #print(filename,name)
    create_spectrogram(filename,name)

img_dir=np.array(glob(root_path+'working/images/*'))
new_image = load_image(img_dir[0])

# check prediction
pred = model.predict(new_image)
#print(pred)
predicted_class = np.argmax(pred,axis=1)
#print(idx2label[predicted_class[0]])
#print(file)
labels_ = []
for i in range(10):
    labels_.append(idx2label[i])

#print(labels_)
#print(pred)
final_prob = list(zip(labels_, pred[0]))
#print(final_prob)
final_prob = sorted(final_prob,key=lambda x: x[1], reverse=True)

#print()
i = 1
for p in final_prob:
    print(str(i)+". " + p[0] + " : " + str(round(p[1]*100,2))+ "%")
    print()
    i+=1    

#print(model.summary())


