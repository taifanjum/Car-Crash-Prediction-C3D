# -*- coding: utf-8 -*-
"""
Created on Fri May 20 18:02:17 2022

@author: Taifanjum
"""

import numpy as np
import cv2
import argparse
import os
#os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '0' # probably 0
import tensorflow as tf
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, ZeroPadding3D)
#from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import videoto3d

#Function for ploting accuracy and loss graph
def plot_history(history, result_dir):
    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()

#Function for loading the data. 
#Takes 5 parameters, the video directory, video frames, number of classes (2 in our case), the directory path to store results..
#color: if we want RGB (colored) images or grayscale
#Skip determines if we take frames at regular intervals (if True) or take the initial frames according to depth (if false) 
def loaddata(video_dir, vid3d, nclass, result_dir, color=False, skip=True):
    files = os.listdir(video_dir) #lists the files in the video directory
    X = [] #Array for frames
    labels = [] # Video labels (Normal or Crash)
    labellist = [] # Video label list

    pbar = tqdm(total=len(files)) #calcualtes the number of videos in the video directory

    for filename in files: #For each video in the video directory 
        pbar.update(1)
        
        name = os.path.join(video_dir, filename) #Joins the video directory path to video name to get the whole path for each video 
        label = vid3d.get_UCF_classname(filename) # gets the filename of the videos and stores them as labels
        if label not in labellist:
            if len(labellist) >= nclass:
                continue
            labellist.append(label) #Add the label to label list
        labels.append(label) #Add the label to labels array
        X.append(vid3d.video3d(name, color=color, skip=skip))

    pbar.close()
    with open(os.path.join(result_dir, 'classes.txt'), 'w') as fp:
        for i in range(len(labellist)):
            fp.write('{}\n'.format(labellist[i]))

    for num, label in enumerate(labellist):
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i] = num
    if color:
        return np.array(X).transpose((0, 2, 3, 4, 1)), labels
    else:
        return np.array(X).transpose((0, 2, 3, 1)), labels


videos = '/Users/Taifanjum/Downloads/CCD/videos/Crash-1500' #Video directory
depth = 16 # number of frames to smaple from each video
nclass = 2 # Number of classes (Normal or Crash)
skip = False #determines if we take frames at regular intervals (if True) or take the initial frames according to depth (if false) 
output = '/Users/Taifanjum/Downloads/CCD/C3D_results/first16_72x128_2' #Direcotry to store the results

if not os.path.isdir(output): #if output direcotry does not exist
    os.makedirs(output) #create the directory 

color = False #if we want RGB (colored) images or grayscale images

img_rows, img_cols, frames = 72, 128, depth #height and width of each image and the number of frames (depth)
channel = 1 #3 if args.color else 1
fname_npz = 'dataset_{}_{}_{}_{}_{}.npz'.format(
    nclass, depth, skip, img_rows, img_cols)
#print(frames)
vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)
nb_classes = nclass
#calling the load data function
if os.path.exists(fname_npz):
    loadeddata = np.load(fname_npz)
    X, Y = loadeddata["X"], loadeddata["Y"]
else:
    x, y = loaddata(videos, vid3d, nclass,
                    output, color, skip)
    X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
    Y = np_utils.to_categorical(y, nb_classes)
    X = X.astype('float32')
    np.savez(fname_npz, X=X, Y=Y)
    print('Saved dataset to dataset.npz.')
print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))

model = Sequential()

    # 1st layer group
model.add(Conv3D(64, (3, 3, 3), activation="relu",name="conv1", 
                 input_shape=(X.shape[1:]),
                 strides=(1, 1, 1), padding="same"))  
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name="pool1", padding="valid"))

# 2nd layer group  
model.add(Conv3D(128, (3, 3, 3), activation="relu",name="conv2", 
                 strides=(1, 1, 1), padding="same"))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool2", padding="valid"))

# 3rd layer group   
model.add(Conv3D(256, (3, 3, 3), activation="relu",name="conv3a", 
                 strides=(1, 1, 1), padding="same"))
model.add(Conv3D(256, (3, 3, 3), activation="relu",name="conv3b", 
                 strides=(1, 1, 1), padding="same"))
model.add(ZeroPadding3D(padding=(0, 1, 1)))	
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool3", padding="valid"))

# 4th layer group  
model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv4a", 
                 strides=(1, 1, 1), padding="same"))   
model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv4b", 
                 strides=(1, 1, 1), padding="same"))
model.add(ZeroPadding3D(padding=(0, 1, 1)))	
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool4", padding="valid"))

# 5th layer group  
model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv5a", 
                 strides=(1, 1, 1), padding="same"))   
model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv5b",
                  strides=(1, 1, 1), padding="same"))
model.add(ZeroPadding3D(padding=(0, 1, 1)))	
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool5", padding="valid"))
model.add(Flatten())
                 
# FC layers group
model.add(Dense(4096, activation='relu', name='fc6'))
model.add(Dropout(.5))
model.add(Dense(4096, activation='relu', name='fc7'))
model.add(Dropout(.5))
model.add(Dense(2, activation='softmax', name='fc8'))

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(), metrics=['accuracy'])
model.summary()
#plot_model(model, show_shapes=True,
        #   to_file=os.path.join(output, 'model.png'))
#split the training data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=43)

#split the training data further to have a validation data
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=43)

#Early stopping is a callback what prevents your model from overfitting by stopping training is the model is not learning for 11 epochs (patience param)
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00008,
    patience=11,
    verbose=1,
    restore_best_weights=True,
)
#Reduces the learning rate by a facotr of 0.25 each time when the model is notl earning for 4 consecutive epochs
lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    min_delta=0.0001,
    factor=0.25,
    patience=4,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [
    early_stopping,
    lr_scheduler,
]
#begin training, pass the training and validation data
history = model.fit(
    X_train, Y_train, batch_size=64, 
    validation_data=(X_val, Y_val), 
    epochs=100, shuffle=True,
    callbacks=callbacks)

#Test the model on the test data
model.evaluate(X_test, Y_test, verbose=0)

#path to save the model for future use
model_path = './model_first16_72x128_2/'

if not os.path.isdir(model_path):
    os.makedirs(model_path)

model.save(model_path)
model_json = model.to_json()

#Saving the model weights, we save the model in multiple ways so we have more flexibility in using the model later on
if not os.path.isdir(output):
    os.makedirs(output)
with open(os.path.join(output, 'crash_3dcnnmodel.json'), 'w') as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(output, 'crash_3dcnnmodel.hd5'))

yhat_test = np.argmax(model.predict(X_test), axis=1)
ytest_ = np.argmax(Y_test, axis=1)

print(classification_report(ytest_, yhat_test))

report = classification_report(ytest_, yhat_test, output_dict=True)

df = pandas.DataFrame(report).transpose()

file = '/Users/Taifanjum/Downloads/CCD/first16_72x128_2.csv'
df.to_csv(file, mode='a')

loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', acc)
plot_history(history, output)
#save_history(history, output)