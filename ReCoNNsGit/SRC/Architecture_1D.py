# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:41:21 2023

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np


dtype="float64"



###Penultimate layer of the NN, used to create the objects neccesary for the 
###Reconn, output of the layer used for explainability
class proto_jump_layer(tf.keras.layers.Layer):
    def __init__(self,):
        super(proto_jump_layer,self).__init__()
    
    def call(self,inputs):
        phi,x = inputs
        
        w1,w2 = tf.unstack(phi,axis=-1)
        
        return w1,w2,x


###Combines the output of the previous layer
class jump_out_layer(tf.keras.layers.Layer):
    def __init__(self,):
        super(jump_out_layer,self).__init__()
    
    def call(self,inputs):
        
        w1,w2,x, = inputs
        
        phi=tf.einsum("ij->i",tf.math.abs(x-np.pi/2))
        
        return w1+w2*phi


###Definition of u_NN for the reconn
def make_jump_model(neurons,activation="tanh"):
    
    init_ker = tf.keras.initializers.GlorotUniform()
    
    xvals = tf.keras.layers.Input(shape=(1,), name="x_input",dtype=dtype)
    
    
    
    l1 = tf.keras.layers.Dense(neurons,activation=activation,dtype=dtype,
                                kernel_initializer = init_ker)(xvals)
    for i in range(2):
        l1 = tf.keras.layers.Dense(neurons,activation=activation,dtype=dtype,name="l"+str(i+2),
                                    kernel_initializer = init_ker)(l1)
    l2 = tf.keras.layers.Dense(2,
                                kernel_initializer = init_ker,dtype=dtype)(l1)
    proto_jump_out = proto_jump_layer()([l2,xvals])
    u_out = jump_out_layer()(proto_jump_out)
    
    
    u_model = tf.keras.Model(inputs=xvals,outputs = u_out)
    
    
    jump_explain_model = tf.keras.Model(inputs=xvals,outputs=proto_jump_out)
    
    u_model.summary()
    
    return u_model, jump_explain_model



##Classical, fully-connected NN.
def make_classic_model(neurons,activation="tanh"):
    
    init_ker = tf.keras.initializers.GlorotUniform()
    
    xvals = tf.keras.layers.Input(shape=(1,), name="x_input",dtype=dtype)
    
    
    
    l1 = tf.keras.layers.Dense(neurons,activation=activation,dtype=dtype,
                                kernel_initializer = init_ker)(xvals)
    for i in range(2):
        l1 = tf.keras.layers.Dense(neurons,activation=activation,dtype=dtype,name="l"+str(i+2),
                                    kernel_initializer = init_ker)(l1)
    l2 = tf.keras.layers.Dense(1,kernel_initializer = init_ker,dtype=dtype)(l1)
    
    
    u_model = tf.keras.Model(inputs=xvals,outputs = l2)
    
    
    u_model.summary()
    
    return u_model
