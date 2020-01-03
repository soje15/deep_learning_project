# -*- coding: utf-8 -*-
"""Fashion_MNIST.ipynb


This code was used in part to simplify the model by focusing only on image
analysis without sequence. However, the result did not improve due to the same
issue as in the main model. The code has been uploaded to show that we put
time into testing this aspect.

"""

import tensorflow as tf
import numpy as np
import keras as ks
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False)


train_generator = datagen.flow_from_directory(
    directory=r"./train/",
    target_size=(720, 720),
    color_mode="rgb",
    batch_size=64,
    class_mode="binary",
    shuffle=True,
    seed=10
)


validation_generator = datagen.flow_from_directory(
    directory=r"./validation/",
    target_size=(720, 720),
    color_mode="rgb",
    batch_size=64,
    class_mode="binary",
    shuffle=True,
    seed=10
)


test_generator = datagen.flow_from_directory(
    directory=r"./test/",
    target_size=(720, 720),
    color_mode="rgb",
    batch_size=64,
    class_mode="binary",
    shuffle=True,
    seed=10
)


# Load the fashion-mnist pre-shuffled train data and test data
#print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)


model = Sequential()
# Must define the input shape in the first layer of the neural network
model.add(Conv2D(filters=64,  kernel_size =(16,16), padding='same', activation='relu', kernel_regularizer  = 'l2', kernel_initializer='glorot_uniform', input_shape=(720,720,3))) 

model.add(ks.layers.BatchNormalization())

model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32,   kernel_size = (8,8), padding='same',  activation='relu'))
model.add(ks.layers.BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.1))


model.add(MaxPooling2D(pool_size=2))


model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

model.summary()


model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


model.fit_generator(train_generator,
         epochs=100,
         steps_per_epoch = 5,
         validation_data=validation_generator

)



def print_perf(model):
            plt.plot(model.history.history['loss'])
            plt.plot(model.history.history['val_loss'])
            plt.title('Fish Validation Loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()
            
print_perf(model)

# Evaluate the model on test set
score = model.evaluate(test_generator, verbose=0)
# Print test accuracy
print('\n', 'Test accuracy:', score[1])