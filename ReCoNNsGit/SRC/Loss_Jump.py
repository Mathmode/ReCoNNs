# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:12:51 2023

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np 

from SRC.Loss_4_materials import stack_layer


dtype="float64"


###Define the material paramters
def sigma(x,y):
    r = (x**2+y**2-1/4)
    ps = (tf.math.sign(r)+1)/2
    return ps+3*(1-ps)


###Exact solution
def sol_exact(x,y):
    return (x**2-1)*(y**2-1)*(4*x**2+4*y**2-1)/sigma(x,y)


###Right-hand side of PDE is defined via the exact solution.
def rhs(x,y):
    with tf.GradientTape(persistent=True) as t1:
        t1.watch(x)
        t1.watch(y)
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(x)
            t2.watch(y)
            u=sol_exact(x,y)
        dux = t2.gradient(u,x)
        duy = t2.gradient(u,y)
    return (t1.gradient(dux,x)+t1.gradient(duy,y))*sigma(x,y)


###The PDE component of the loss. 
class loss_layer_col_sq(tf.keras.layers.Layer):
    def __init__(self,u_model,n,**kwargs):
        super(loss_layer_col_sq,self).__init__()
        self.u_model=u_model
        self.n=n
    
    def call(self,inputs):
        ###MC Sample
        x = tf.random.uniform([self.n],minval=-1,dtype=dtype)
        y=tf.random.uniform([self.n],minval=-1,dtype=dtype)
        
        
        ###Evaluate derivatives
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            t1.watch(y)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x)
                t2.watch(y)
                u = self.u_model(tf.stack([x,y],axis=-1))
            dux = t2.gradient(u,x)
            duy = t2.gradient(u,y)
        duxx = t1.gradient(dux,x)
        duyy = t1.gradient(duy,y)
        
        ##Evaluate residual and loss
        f = rhs(x,y)
        lap = sigma(x,y)*(duxx+duyy)
        
        ans = tf.reduce_mean(tf.square((lap-f)))*4
        
        
        return ans


###Loss component corresponding to the Dirichlet boundary condition,
class loss_layer_dirichlet_sq(tf.keras.layers.Layer):
    def __init__(self,u_model,n,**kwargs):
        super(loss_layer_dirichlet_sq,self).__init__()
        self.n= int(n/4)
        self.u_model=u_model
        
    def call(self,inputs):
        
        ###Sampling on each edge and evaluation of loss on each edge
        x1 = tf.random.uniform([self.n],dtype=dtype,minval=-1)
        one = tf.ones_like(x1)
        
        u1 = tf.reduce_mean(self.u_model(tf.stack([one,x1],axis=-1))**2)*2
        
        x1 = tf.random.uniform([self.n],dtype=dtype,minval=-1)
        u2 = tf.reduce_mean(self.u_model(tf.stack([-one,x1],axis=-1))**2)*2
        
        x1 = tf.random.uniform([self.n],dtype=dtype,minval=-1)
        u3 = tf.reduce_mean(self.u_model(tf.stack([x1,one],axis=-1))**2)*2
        
        x1 = tf.random.uniform([self.n],dtype=dtype,minval=-1)
        u4 = tf.reduce_mean(self.u_model(tf.stack([x1,-one],axis=-1))**2)*2       
        
        
        ##Evaluate loss
        ans= u1+u2+u3+u4
        return ans


##Component of the loss corresponding to the continuity of the flux
class loss_layer_jump(tf.keras.layers.Layer):
    def __init__(self,jump_explain_model,n):
        super(loss_layer_jump,self).__init__()
        self.j_model = jump_explain_model
        self.n = n
    
    def call(self,inputs):
        
        
        ###Sample on the interface via polar coordinates
        t0 = tf.random.uniform([self.n],dtype=dtype,maxval=2*np.pi)
        x= tf.math.cos(t0)*1/2
        y = tf.math.sin(t0)*1/2
        
        ##Unit vector normal to the surface
        nu = tf.stack([2*x,2*y],axis=-1)
        
        
        ##Evaluate derivatives
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            t1.watch(y)
            xy = tf.stack([x,y],axis=-1)
            w1,w2,fx,fy = self.j_model(xy)
        
        ##Evaluate normal derivative of the smooth part of the solution
        gw = tf.stack([t1.gradient(w1,x),t1.gradient(w1,y)],axis=-1)
        
        ##Calculate the jump component of the derivative
        jw = tf.einsum("I,Ii->Ii",w2,2*xy)
        
        ##Left- and right- derivatives
        inside = gw-jw
        outside = gw+jw
        
        ##Evaluate loss
        return tf.reduce_mean(tf.einsum("Ii,Ii->I",3*inside-1*outside,nu)**2)*np.pi
        


###Define the full loss function. 
def make_loss_jump(u_model,jump_explain_model,n):
    xvals = tf.keras.Input(shape=(1,),dtype=dtype)
    bc_loss= loss_layer_dirichlet_sq(u_model,n)(xvals)
    jump_loss = loss_layer_jump(jump_explain_model,n)(xvals)
    pinns_loss = loss_layer_col_sq(u_model,n)(xvals)
    
    out = stack_layer()([pinns_loss,bc_loss,jump_loss])
    
    loss_model = tf.keras.Model(inputs=xvals,outputs=out)
    return loss_model

