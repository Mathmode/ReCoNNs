# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:12:51 2023

@author: jamie.taylor
"""


import tensorflow as tf
import numpy as np


dtype="float64"



###Penultimate layer of the network - used to create the necessary
###components of the ReCoNN and for explainability in the loss
class proto_jump_layer(tf.keras.layers.Layer):
    def __init__(self,):
        super(proto_jump_layer,self).__init__()
    
    def call(self,inputs):
        phi,xy = inputs
        
        w1,w2 = tf.unstack(phi,axis=-1)
        
        
                
        x,y = tf.unstack(xy,axis=-1)
        return w1,w2,x,y
    
    
###Combine the elements from the previous layer
class jump_out_layer(tf.keras.layers.Layer):
    def __init__(self,):
        super(jump_out_layer,self).__init__()
    
    def call(self,inputs):
        
        w1,w2,x,y = inputs
        
        phi=tf.math.abs(x**2+y**2-1/4)
        
        return w1+w2*phi


###Full definition of the ReCoNN,
def make_jump_model(neurons,activation="tanh",bc=True):
    
    init_ker = tf.keras.initializers.GlorotUniform()
    
    xvals = tf.keras.layers.Input(shape=(2,), name="x_input",dtype=dtype)
    
    l2w = 0.
    
    
    l1 = tf.keras.layers.Dense(neurons,activation=activation,dtype=dtype,
                                kernel_initializer = init_ker,
                                kernel_regularizer=tf.keras.regularizers.L2(l2w))(xvals)
    for i in range(2):
        l1 = tf.keras.layers.Dense(neurons,activation=activation,dtype=dtype,name="l"+str(i+2),
                                    kernel_initializer = init_ker,
                                    kernel_regularizer=tf.keras.regularizers.L2(l2w))(l1)
    l2 = tf.keras.layers.Dense(2,
                                kernel_initializer = init_ker,dtype=dtype)(l1)
    proto_jump_out = proto_jump_layer()([l2,xvals])
    u_out = jump_out_layer()(proto_jump_out)
    
    
    u_model = tf.keras.Model(inputs=xvals,outputs = u_out)
    
    
    jump_explain_model = tf.keras.Model(inputs=xvals,outputs=proto_jump_out)
    
    u_model.summary()
    
    return u_model, jump_explain_model
