import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import csv
from tensorflow.keras.utils import to_categorical


'''
--------------------------------------------
Engage GPU
--------------------------------------------
'''

"""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
"""

conversion = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
              'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14,
              'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
              'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27,
              'space': 28}

Y =  np.array([[]])
with open('landmark.csv') as csv_file:

    csv_reader = csv.reader(csv_file, delimiter=' ')

    datalength = 63119

    X = np.zeros((datalength, 42))  #  21 points, each with x,y
    row_count = 0
    for row in csv_reader:  # for every image
        Y_value = conversion[row[0]]
        Y = np.append(Y, [Y_value])
        input = row[1:43]
        for num in range(len(input)):
            X[row_count,num] = input[num]
        row_count += 1


indexs = list(range(0, datalength, 5)) # Every 5th letter is a validation image (80-20 split)

vallength = len(indexs) # Length of validation set
X_val = np.zeros((vallength, 42)) # Initialise numpy array first
Y_val = np.array([[]])


# Add the correct validation data to X_Val and Y_val
for num in range(vallength):
    index = indexs[num]
    X_coords = X[index,:]

    for coordnum in range(len(X_coords)):
        X_val[num,coordnum] = X_coords[coordnum]

    Y_val = np.append(Y_val, Y[index])


# del indexes from Y and X (removes val data from train data)
Y_train = np.delete(Y, indexs)
X_train = np.delete(X, indexs, axis = 0)


Y_train_cat = to_categorical(Y_train, 29)
Y_val_cat = to_categorical(Y_val, 29)


train_dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train_cat)).batch(30)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val,Y_val_cat)).batch(30)

'''
--------------------------------------------
Training the model
--------------------------------------------
'''

print('Started training...')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience = 2)

model = Sequential([
        Flatten(input_shape = (42,1) ),
        Dense(units = 47, activation = 'tanh'),
        Dense(units = 29, activation = 'softmax')
        ])


model.compile(optimizer = Adam(learning_rate = 0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x = train_dataset,
          batch_size = 30,
          epochs = 99999,
          shuffle = False,  # Already set to true
          verbose = 2,  # See amount of output messages, 2 = highest, 0 = lowest
          validation_data = val_dataset,
          callbacks = [early_stopping_monitor]
          )

import os.path
if os.path.isfile('test.h5') is False:
    model.save('test.h5')