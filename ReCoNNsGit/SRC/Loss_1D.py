# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:41:03 2023

@author: jamie.taylor
"""



import tensorflow as tf

import numpy as np


from SRC.Loss_4_materials import stack_layer

dtype="float64"



###Define functions for the exact solution. 
def sigma(x):
    ps = (tf.math.sign(x-np.pi/2)+1)/2
    return ps+3*(1-ps)

def sol_exact(x):
    return tf.math.sin(2*x)/sigma(x)

def rhs(x):
    return tf.math.sin(2*x)*(-4)


##PINNs PDE loss 
class loss_layer_col_sq(tf.keras.layers.Layer):
    def __init__(self,u_model,n,**kwargs):
        super(loss_layer_col_sq,self).__init__()
        self.u_model=u_model
        self.n=n
    
    def call(self,inputs):
        
        ##Sampling 
        x = tf.random.uniform([self.n],maxval=np.pi,dtype=dtype)
        
        ##evaluation of derivatives
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x)
                u = self.u_model(x)
            dux = t2.gradient(u,x)
        duxx = t1.gradient(dux,x)
        f = rhs(x)
        lap = sigma(x)*(duxx)
        
        ans = tf.reduce_mean(tf.square((lap-f)))
        
        
        tf.print("Col = ",ans)
        return ans

##Component of the loss for Dirichlet BC
class loss_layer_dirichlet_sq(tf.keras.layers.Layer):
    def __init__(self,u_model,**kwargs):
        super(loss_layer_dirichlet_sq,self).__init__()
        self.u_model=u_model
        self.x=tf.constant([0.,np.pi],dtype=dtype)
        
    def call(self,inputs):
        return tf.reduce_sum(self.u_model(self.x)**2)
    
    
##Component of the loss for the interface condition
class loss_layer_jump(tf.keras.layers.Layer):
    def __init__(self,jump_explain_model):
        super(loss_layer_jump,self).__init__()
        self.j_model = jump_explain_model
        self.sigma = sigma
        self.x = tf.constant([np.pi/2],dtype=dtype)
    
    def call(self,inputs):
        
        ##Evalute derivative of the smooth part of u_NN
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(self.x)
            w1,w2,x = self.j_model(self.x)
        gw = t1.gradient(w1,self.x)
        
        ##Evaluate the left- and right- derivatives
        inside = gw-w2
        outside = gw+w2
        return tf.reduce_mean((3*inside-1*outside)**2)


###Definition of full loss function
def make_loss_jump(u_model,jump_explain_model,n):
    xvals = tf.keras.Input(shape=(1,),dtype=dtype)
    bc_loss= loss_layer_dirichlet_sq(u_model)(xvals)
    jump_loss = loss_layer_jump(jump_explain_model)(xvals)
    pinns_loss = loss_layer_col_sq(u_model,n)(xvals)
    
    
    out = stack_layer()([pinns_loss,bc_loss,jump_loss])
    loss_model = tf.keras.Model(inputs=xvals,outputs=out)
    return loss_model



###Defines the MC loss for using H^1 norm as a loss function. 
class mc_loss_layer(tf.keras.layers.Layer):
    def __init__(self,u_model,n,**kwargs):
        super(mc_loss_layer,self).__init__()
        self.u_model = u_model
        self.n = n
    
    def call(self,inputs):
        x=tf.random.uniform([self.n],maxval=np.pi,dtype=dtype)
        with tf.GradientTape() as t1:
            t1.watch(x)
            uerr = tf.squeeze(self.u_model(x))-sol_exact(x)
        duerr = t1.gradient(uerr,x)
        return tf.reduce_mean(uerr**2+duerr**2)

   
def make_loss_MC(u_model,n):
    xvals = tf.keras.Input(shape=(1,),dtype=dtype)
    out = mc_loss_layer(u_model,n)(xvals)
    loss_model = tf.keras.Model(inputs=xvals,outputs=out)
    return loss_model



            