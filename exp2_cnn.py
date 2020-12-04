# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:03:34 2020
@author: Cleiton Moya de Almeida
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

# 1. Data Loading and Pre-Processing
# Dataset loading
(x_train, y_train), (x_val, y_val) = mnist.load_data()

# Normalization
x_train = x_train / 255
x_val = x_val / 255

# Reshaping
img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)

# Label encoding (one hot encoding)
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

# 2. Model design
model_1 = Sequential()
model_1.add(Conv2D(32, kernel_size=(3, 3), 
     activation='relu', input_shape=(img_rows, img_cols, 1)))
model_1.add(MaxPooling2D(pool_size=(2, 2)))
model_1.add(Dropout(0.25))
model_1.add(Flatten())
model_1.add(Dense(num_classes, activation='softmax'))

model_1.compile(loss = 'mean_squared_error',
      optimizer='sgd',
      metrics=['accuracy'])

model_2 = Sequential()
model_2.add(Conv2D(32, kernel_size=(3, 3), 
     activation='relu', input_shape=(img_rows, img_cols, 1)))
model_2.add(MaxPooling2D(pool_size=(2, 2)))
model_2.add(Dropout(0.25))
model_2.add(Flatten())
model_2.add(Dense(num_classes, activation='softmax'))

model_2.compile(loss = 'categorical_crossentropy',
      optimizer='sgd',
      metrics=['accuracy'])

model_3 = Sequential()
model_3.add(Conv2D(32, kernel_size=(3, 3), 
     activation='relu', input_shape=(img_rows, img_cols, 1)))
model_3.add(MaxPooling2D(pool_size=(2, 2)))
model_3.add(Dropout(0.25))
model_3.add(Flatten())
model_3.add(Dense(num_classes, activation='softmax'))

model_3.compile(loss = 'categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])

model_4 = Sequential()
model_4.add(Conv2D(32, kernel_size=(3, 3),
                   activation='relu', input_shape=(img_rows, img_cols, 1)))
model_4.add(MaxPooling2D(pool_size=(2, 2)))
model_4.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model_4.add(MaxPooling2D(pool_size=(2, 2)))
model_4.add(Dropout(0.25))
model_4.add(Flatten())
model_4.add(Dense(num_classes, activation='softmax'))
model_4.compile(loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])

model_5 = Sequential()
model_5.add(Conv2D(32, kernel_size=(3, 3), 
     activation='relu', input_shape=(img_rows, img_cols, 1)))
model_5.add(MaxPooling2D(pool_size=(2, 2)))
model_5.add(Conv2D(64, kernel_size=(3, 3), activation='relu',kernel_regularizer=l2(5e-4)))
model_5.add(MaxPooling2D(pool_size=(2, 2)))
model_5.add(Dropout(0.25))
model_5.add(Flatten())
model_5.add(Dense(num_classes, activation='softmax'))
model_5.compile(loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])

model_6 = Sequential()
model_6.add(Conv2D(64, kernel_size=(3, 3), kernel_regularizer=l2(5e-4),
     activation='relu', input_shape=(img_rows, img_cols, 1)))
model_6.add(MaxPooling2D(pool_size=(2, 2)))
model_6.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model_6.add(MaxPooling2D(pool_size=(2, 2)))
model_6.add(Dropout(0.25))
model_6.add(Flatten())
model_6.add(Dense(num_classes, activation='softmax'))
model_6.compile(loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])

model_7 = Sequential()
model_7.add(Conv2D(64, kernel_size=(3, 3),
     activation='relu', input_shape=(img_rows, img_cols, 1)))
model_7.add(MaxPooling2D(pool_size=(2, 2)))
model_7.add(Conv2D(64, kernel_size=(3, 3), activation='relu',kernel_regularizer=l2(5e-4)))
model_7.add(MaxPooling2D(pool_size=(2, 2)))
model_7.add(Dropout(0.25))
model_7.add(Flatten())
model_7.add(Dense(num_classes, activation='softmax'))
model_7.compile(loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])

# Model selection
model = model_6

# Experiment design
batch_size = 64
epochs = 5
early_stop = EarlyStopping(monitor='loss', patience=1)

# Trainning and evaluating
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val),
          callbacks=[early_stop])

score = model.evaluate(x_val, y_val, verbose=0)
print('val loss:', score[0])
print('val accuracy:', score[1])

# Plottint the results
plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(2,1,figsize=(4,3))
ax[0].plot(history.history['accuracy'], label="Training")
ax[0].plot(history.history['val_accuracy'],label="Validation")
ax[0].set_title('Accuracy')
ax[0].grid(linestyle='--')
ax[0].set_xticks(np.arange(0,epochs))
ax[0].set_xticklabels(np.arange(1,1+epochs))
ax[0].legend()

ax[1].plot(history.history['loss'], label="Training")
ax[1].plot(history.history['val_loss'], label="Validation")
ax[1].set_title('Loss')
ax[1].grid(linestyle='--')
ax[1].set_xticks(np.arange(0,epochs))
ax[1].set_xticklabels(np.arange(1,1+epochs))
ax[1].set_xlabel('Epochs')
fig.tight_layout()