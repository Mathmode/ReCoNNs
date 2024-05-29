# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:42:30 2023

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np
from SRC.Architecture_4_materials import relcut
from SRC.Loss_4_materials import relcut_C0, stack_layer

dtype="float64"
tf.keras.backend.set_floatx(
    dtype
)


###Definition of factor in the weighting function for PDE loss
delta1 = tf.constant(0.025,dtype=dtype)

##Exponent in the weighting function of PDE loss
exponent_bulk=2.

pi = tf.constant(np.pi,dtype=dtype)


##MC sample in the L-shaped domain
def sample_L_random(n,maxval=1):
    m=int(n/3)
    x1 = tf.random.uniform([m],dtype=dtype)*maxval
    y1 = tf.random.uniform([m],dtype=dtype)*maxval
    x2 = tf.random.uniform([m],dtype=dtype)*maxval-maxval
    y2 = tf.random.uniform([m],dtype=dtype)*maxval
    x3 = tf.random.uniform([m],dtype=dtype)*maxval
    y3 = tf.random.uniform([m],dtype=dtype)*maxval-maxval
    return tf.concat([x1,x2,x3],axis=-1),tf.concat([y1,y2,y3],axis=-1)

###Definition of the singular part of the solution
def singular_sol(x,y):
    pot = (x**2+y**2)**(1./3.)
    return  pot*tf.math.sin((2/3)*(tf.math.atan2(x,y)+pi/2))


##Exact solution
def sol_exact(x,y):
    singular_part = singular_sol(x,y)
    return singular_part*(x-1)*(x+1)*(y-1)*(y+1)


##Right-hand side of the PDE defined via autodiff
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
    return t1.gradient(dux,x)+t1.gradient(duy,y)


###PDE part of the loss
class loss_layer_col_L(tf.keras.layers.Layer):
    def __init__(self,u_model,n,**kwargs):
        super(loss_layer_col_L,self).__init__()
        self.u_model=u_model
        self.n=n
        self.delta= delta1
    
    def call(self,inputs):
        
        ##Sample points
        x,y = sample_L_random(self.n)
        
        ##Evaluate all derivatives
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
        f = rhs(x,y)
        
        ##Weighting function
        r=tf.sqrt((x**2+y**2))
        omega=relcut_C0(r/self.delta)**exponent_bulk
        
        
        ##Residual
        ans = tf.reduce_mean(tf.square((duxx+duyy-f))*omega)*3
        
        return ans



##Defin the BC part of the loss for the angular function
class loss_layer_dirichlet_L_sing(tf.keras.layers.Layer):
    def __init__(self,singular_explain,**kwargs):
        super(loss_layer_dirichlet_L_sing,self).__init__()
        self.singular_explain=singular_explain
        
    def call(self,inputs):
        phi,radcut,s,r = self.singular_explain(tf.constant([[-0.5,0],[0,-0.5]],dtype=dtype))
        return tf.reduce_sum(phi**2)
    
##Defines the BC part of the loss for u_NN
class loss_layer_dirichlet_L(tf.keras.layers.Layer):
    def __init__(self,u_model,n,**kwargs):
        super(loss_layer_dirichlet_L,self).__init__()
        self.n= int(n/8)
        self.u_model=u_model
        
    def call(self,inputs):
        
        ##Sampling over each edge and combination at the end. 
        x1 = tf.random.uniform([self.n],dtype=dtype)
        zeros = tf.zeros_like(x1)
        ones = tf.ones_like(x1)
        p1 = tf.reduce_mean(self.u_model(tf.stack([zeros,-x1],axis=-1))**2)
        
        x1 = tf.random.uniform([self.n],dtype=dtype)
        ones = tf.ones_like(x1)
        p2 = tf.reduce_mean(self.u_model(tf.stack([-x1,zeros],axis=-1))**2)
        
        x1 = tf.random.uniform([2*self.n],dtype=dtype)
        ones = tf.ones_like(x1)
        p3 = tf.reduce_mean(self.u_model(tf.stack([-ones,2*x1-1],axis=-1))**2)*2
        
        x1 = tf.random.uniform([self.n],dtype=dtype)
        ones = tf.ones_like(x1)
        p4 = tf.reduce_mean(self.u_model(tf.stack([2*x1-1,ones],axis=-1))**2)
        
        x1 = tf.random.uniform([2*self.n],dtype=dtype)
        ones = tf.ones_like(x1)
        p5 = tf.reduce_mean(self.u_model(tf.stack([ones,2*x1-1],axis=-1))**2)*2
        
        x1 = tf.random.uniform([self.n],dtype=dtype)
        ones = tf.ones_like(x1)
        p6 = tf.reduce_mean(self.u_model(tf.stack([x1,-ones],axis=-1))**2)
        
        
        ans= (p1+p2+p3+p4+p5+p6)
        
        return ans


##Full loss function, singular_explain=False does not include the 
##loss for the BC of the angular function, used for classical architecture.
def make_loss_L(u_model,n,singular_explain=False):
    xvals = tf.keras.Input(shape=(1,),dtype=dtype)
    bc_loss= loss_layer_dirichlet_L(u_model,n)(xvals)
    pinns_loss = loss_layer_col_L(u_model,n)(xvals)
    if singular_explain ==False:
        out = stack_layer()([pinns_loss,bc_loss])
    else:
        bc_sing_loss = loss_layer_dirichlet_L_sing(singular_explain)(xvals)
        out = stack_layer()([pinns_loss,bc_loss,bc_sing_loss])
    
    loss_model = tf.keras.Model(inputs=xvals,outputs=out)
    return loss_model