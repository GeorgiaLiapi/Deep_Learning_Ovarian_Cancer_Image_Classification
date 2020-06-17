# -*- coding: utf-8 -*-
"""
        Author: Georgia Liapi, Master Student in Systems Biology, Maastricht University
 Academic Year: 2019-2020
       Project: 'Deep learning-based prediction of molecular subtypes in human ovarian cancer'
       Utility: Selects and loads the finalized model structure on the Train_Val_Pred.py or the Train_whole_Test.py script
"""
# Import required libraries
from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.optimizers import Adam
from keras.preprocessing.image import image
from keras.regularizers import l2
from keras.models import Sequential
from keras.utils import layer_utils
from keras import regularizers
import ipykernel
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K

# IGNORE ANY WARNING MESSAGES- THEY DO NOT AFFECT THE PROCESS

# Supress deprecation messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def Set_Network(network , num_classes):
     
     # Clear previous sessions
     K.clear_session()

     # Set a seed for reproducibility
     tf.compat.v1.set_random_seed(1234)
      
     # Configure the current session
     Configuration = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
     Session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=Configuration)
     K.set_session(Session)
     
     # Set parameters (weights and bias regularizer)
     regl2=l2(0.0001)
     
     # ********************* Xception *****************************************
     if network == 'Xception':
          
          input_tensor= Input(shape=(299,299,3))

          # Import the base pre-trained network
          xception= Xception(include_top=False, weights= 'imagenet', pooling=None, input_tensor= input_tensor)
               
          # Create a new Keras Sequential model and firstly add the base model
          model = Sequential()
          model.add(xception)          
          model.add(GlobalMaxPooling2D())
          model.add(Dropout(rate=0.3))
          model.add(Dense(55, activation='relu', kernel_regularizer=regl2, 
                                                 bias_regularizer=regl2))
          model.add(Dropout(rate=0.5))
          model.add(Dense(num_classes, activation='softmax'))
          
          # Keep the base layers frozen
          for layer in xception.layers: 
                layer.trainable = False         
                             
     # ********************* Resnet50 *****************************************     
     elif network == 'ResNet50':
          
          input_tensor= Input(shape=(224,224,3))

          # Import the base pre-trained network
          resnet = ResNet50(include_top=False, weights='imagenet', pooling=None, input_tensor= input_tensor)

          # Create a new Keras Sequential model and firstly add the base model
          model = Sequential()
          model.add(resnet)
          model.add(GlobalMaxPooling2D())
          model.add(Dropout(rate=0.4))
          model.add(Dense(55, activation='relu',kernel_regularizer=regl2,
                                                bias_regularizer=regl2))
          model.add(Dropout(rate=0.6))
          model.add(Dense(num_classes, activation='softmax'))
              
          # Keep the base layers frozen
          for layer in resnet.layers:
               layer.trainable = False    
                                              
     # ********************* Inceptionv3  *************************************
     elif network == 'InceptionV3':
          
          input_tensor= Input(shape=(299,299,3))
          
          # Import the base pre-trained network
          inceptionv3 = InceptionV3(include_top=False, weights='imagenet', pooling=None, input_tensor= input_tensor)

          # Create a new Keras Sequential model and firstly add the base model
          model = Sequential()
          model.add(inceptionv3)
          model.add(GlobalAveragePooling2D())
          model.add(Dropout(rate=0.4))
          model.add(Dense(100, activation='relu', kernel_regularizer=regl2,
                                                 bias_regularizer=regl2))
          model.add(Dropout(rate=0.7))
          model.add(Dense(num_classes, activation='softmax'))
                              
          # Keep the base layers frozen
          for layer in inceptionv3.layers:
               layer.trainable = False
                       
          
     return model;
