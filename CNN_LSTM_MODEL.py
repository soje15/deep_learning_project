#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import keras as ks
from keras import regularizers
import keras.preprocessing.image as img
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout, LSTM
from collections import deque
from keras.layers.wrappers import TimeDistributed
#from keras.layers import Dropout
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import cv2

path = '/work1/s193416/videos/Bredgrund Videos/experiment/small_data/'

from keras_video_backup.generator import VideoFrameGenerator 

from keras.preprocessing.image import ImageDataGenerator


def colorspace(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

#Image transformer, to do fun stuff
    
image_transformer = ImageDataGenerator(
#rescale= 1 / 5,
# rotation_range=20,
# width_shift_range=0.2,
# height_shift_range=0.2,
# horizontal_flip=True,
 #preprocessing_function = colorspace
 )


#nb_frames = 90 gives the model 3 seconds of video assuming 30 fps. 

train_generator = VideoFrameGenerator(nb_frames = 3, split = 0.5, nb_channel =3, target_shape = (256, 144) , batch_size = 10, transformation = image_transformer , classes = ['0','1'],glob_pattern = '/work1/s193416/videos/Bredgrund Videos/data2/{classname}/*.LRV')
validation_generator = train_generator.get_validation_generator()
image, label = train_generator.__getitem__(5)
image.shape


class CNN_LSTM_Model():
    
      def __init__(self, train_generator , validation_generator):
        
        self.n_epochs = 100
    
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        #self.input_shape = (2, 2 ,256, 256, 3)
          
        #data params  
        self.input_shape = (self.train_generator.batch_size, self.train_generator.nbframe ,self.train_generator.target_shape, self.train_generator.nb_channel)
        self.C = self.train_generator.classes_count
        
        
        #Parameters for the RNNs (STANDALONE - NOT USED)
        self.LSTM_one_cells = 128
        self.LSTM_two_cells = 128
        
        
        #optimization
        self.weight_init = 'glorot_uniform'
        self.regularizer_lambda  = 0.001
        
        
       # self.opt = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
        self.loss_func ='categorical_crossentropy'
        self.opt = tf.keras.optimizers.Adam(lr = 1e-3, decay = 1e-6)
        
        
        
        # Parameters for the dense layers
        self.nodes_dense_layer_one = 64
        
        #Sequence specific params
        #self.len_seq = 50
        
        
        #from init cnn
        self.layer_one_filters = 64
        self.layer_one_filter_size = (8, 8)
        self.pool_one_size = (4, 4)
        self.dropout_one_probability = 0.75 # 0.5
        
        
        self.model = self.init_CNN_RNN()
        
    

      def init_CNN(self):
        

        
        """
        We are using a sequential model for our project
        A sequential model builds up the layer like stacks or in a linear manner
        """
        
        fish_cnn_model = ks.Sequential() # Our CNN model
        print('CNN created')
        

        """
        Here onwards, we are add a layer at a time to our CNN
        """
        
        # Adding the first layer
        fish_cnn_model.add(TimeDistributed(Conv2D(self.layer_one_filters, self.layer_one_filter_size,
                                  activation = 'relu', input_shape = self.input_shape)))
        
        # Adding a pooling layer
        fish_cnn_model.add(TimeDistributed(MaxPooling2D(self.pool_one_size)))
        
        # Using dropout
        fish_cnn_model.add(Dropout(self.dropout_one_probability))
        return fish_cnn_model
    
    
      
        
      def init_RNN(self):
          
            fish_rnn_model = Sequential()
            
            # First we add the LSTM layer with 128 cells.
            # The return_sequence argument tells us if we want our layer to return sequences or not
            # If we are connecting our layer to a dense layer we don't need sequences, put that as False
            # Otherwise, it is true
            fish_rnn_model.add(LSTM(units = 32, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
            fish_rnn_model.add(Dropout(0.5))
            
            fish_rnn_model.add(LSTM(units = 32, activation='tanh', recurrent_activation='sigmoid', return_sequences = False))
            fish_rnn_model.add(Dropout(0.5))
            
            fish_rnn_model.add(Dense(self.nodes_dense_layer_one, activation = 'relu'))
            fish_rnn_model.add(Dropout(0.5))
            
            fish_rnn_model.add(Dense(self.C, activation = 'softmax'))
            
            print('RNN created')
            
            return fish_rnn_model
        

      def train(self):
          
            self.model.compile(optimizer=self.opt, loss = self.loss_func)
            
            print('-------------Fitting mode ; - ) l------------------')
            
            
            self.model.fit(
            x = self.train_generator[0][0],
            y = self.train_generator[0][1])
            #self.model.fit_generator(self.train_generator)
            
            self.model.fit_generator(self.train_generator,
                    steps_per_epoch= 1,
                    validation_data = self.validation_generator,
                   # validation_steps=5, 
                   # verbose = 1,
                    workers =1, #More workers = more batching between cores
                    epochs=self.n_epochs)
            
            print('-------------model summary:------------------')
            
            print(self.model.summary())
            
            print('-------------model output:------------------')
            
            print(self.model.output)
            
            
            return self.model
            
            

            
      def init_CNN_RNN(self):
            print('Initializing model')
            
            """ FROM KERAS DOC:
                
            Consider a batch of 32 samples, where each sample is a sequence of 10 vectors of 16 dimensions.
            The batch input shape of the layer is then (32, 10, 16), and the input_shape, not including the
            samples dimension, is (10, 16).
            
            You can then use TimeDistributed to apply a Dense layer to each of the 10 timesteps,
            independently:
            """
            

            CNN_LSTM = Sequential()
            
    
            # first (non-default) block
           # print('stride 1')
            CNN_LSTM.add(TimeDistributed(Conv2D(32, (16,16), activation = 'relu', input_shape = self.input_shape, strides = (2,2))) )
           # print('strides2')
            CNN_LSTM.add(TimeDistributed(ks.layers.BatchNormalization()))
            
            CNN_LSTM.add(TimeDistributed(ks.layers.Activation('relu')))
            
            CNN_LSTM.add(TimeDistributed(Conv2D(64, (8,8), kernel_initializer=self.weight_init, )))
            CNN_LSTM.add(TimeDistributed(ks.layers.BatchNormalization()))
            CNN_LSTM.add(TimeDistributed(ks.layers.Activation('relu')))
            
            #print('stride 2')
            CNN_LSTM.add(TimeDistributed(MaxPooling2D((8, 8), strides =(2,2)) ))
        
            # LSTM output head
            CNN_LSTM.add(TimeDistributed(ks.layers.Flatten()))
            CNN_LSTM.add(LSTM(32, return_sequences=False, dropout=0.5))
            CNN_LSTM.add(Dense(self.C, activation='softmax'))

            
            print('Model initialized')

            return CNN_LSTM
    
    
cnnrnn = CNN_LSTM_Model(train_generator, validation_generator)

model = cnnrnn.train()

model._feed_outputs

def print_perf(model):
            plt.plot(model.history.history['loss'])
            plt.plot(model.history.history['val_loss'])
            plt.title('Fish Validation Loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()
            
print_perf(model)