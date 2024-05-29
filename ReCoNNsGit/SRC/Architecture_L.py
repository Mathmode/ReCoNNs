# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:42:02 2023

@author: jamie.taylor
"""


import tensorflow as tf
import numpy as np
from SRC.Architecture_4_materials import lamfun, relcut
dtype="float64"
tf.keras.backend.set_floatx(
    dtype
)








###Values for the smooth cutoff function. 
delta1=tf.constant(0.5,dtype=dtype)
delta2 = tf.constant(0.9,dtype=dtype)





###Layer mapping X to X/||X|| to make outputs depend only on the angle. 
class angle_layer(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(angle_layer,self).__init__()
        
    
    def call(self,inputs):
        xy = inputs
        
        r = tf.sqrt(tf.reduce_sum(xy**2,axis=-1))
        
        ang=tf.einsum("ij,i->ij",xy,r**-1)
        return ang


##Mapping X to ||X||
class r_layer(tf.keras.layers.Layer):    
    def __init__(self,**kwargs):
        super(r_layer,self).__init__()
    
    def call(self,inputs):
        return tf.sqrt(tf.reduce_sum(inputs**2,axis=-1))
        


###Penultimate layer of the singular part. Assembles the cutoff function, 
###and contains the singular exponent. Output used for explinability. 
class proto_singular_layer(tf.keras.layers.Layer):
    def __init__(self,):
        super(proto_singular_layer,self).__init__()
        self.delta1=delta1
        self.delta2=delta2
        self.a = tf.constant(-1/(self.delta1 - self.delta2),dtype=dtype)
        self.b = tf.constant(self.delta1/(self.delta1 - self.delta2),dtype=dtype)
        
    def build(self,input_shape):
        laminit = tf.keras.initializers.RandomNormal(mean=0.5,stddev=0.0)
        self.lam = self.add_weight(shape=[1],dtype=dtype,initializer=laminit)
        
    def call(self,inputs):
        r,phi,angs=inputs
        radcut = relcut(self.a*r+self.b)  
        phi = tf.einsum('ij->i',phi)
        
        return phi,radcut,lamfun(self.lam[0]),r

###Assembles outputs of the previous layer. 
class  singular_out_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(singular_out_layer,self).__init__()
    def call(self,inputs):
        z,xy = inputs
        phi,radcut,lam,r =z
        
        
        u = phi*radcut*(r**lam)

        return u

###Last layer of the smooth part. Assembles the cutoff function, 
###Output used for explinability. 
class proto_jump_layer(tf.keras.layers.Layer):
    def __init__(self,):
        super(proto_jump_layer,self).__init__()
        
        self.delta1=delta1
        self.delta2=delta2
        self.a = -1/(self.delta1 - self.delta2)
        self.b = self.delta1/(self.delta1 - self.delta2)
    
    def call(self,inputs):
        phi,xy,r = inputs
        
        
        p1,p2 = tf.unstack(phi,axis=-1)
        
        cut = relcut(self.a*r+self.b)
        
        ph1 = p1+cut*p2
                
        return ph1
    

###Creates the ReCoNN
def make_L_model(neurons,neurons2,activation="tanh"):
    
    init_ker = tf.keras.initializers.GlorotUniform()
    
    xvals = tf.keras.layers.Input(shape=(2,), name="x_input",dtype=dtype)
    rvals = r_layer()(xvals)
    angvals = angle_layer()(xvals)
    
    ###Definition of layers for the singular part of the solution. 
    tl1 = tf.keras.layers.Dense(neurons2,activation=activation,dtype=dtype,
                                kernel_initializer = init_ker)(angvals)
    for i in range(2):
        tl1 = tf.keras.layers.Dense(neurons2,activation=activation,dtype=dtype,
                                    kernel_initializer = init_ker)(tl1)

    tl4 = tf.keras.layers.Dense(1,dtype=dtype,
                                kernel_initializer = init_ker)(tl1)
    p_sing = proto_singular_layer()([rvals,tl4,angvals])
    sing_out = singular_out_layer()([p_sing,angvals])
    
    ###Definition of layers for the smooth part of the solution. 
    l1 = tf.keras.layers.Dense(neurons,activation=activation,dtype=dtype,
                                kernel_initializer = init_ker)(xvals)
    for i in range(2):
        l1 = tf.keras.layers.Dense(neurons,activation=activation,dtype=dtype,name="l"+str(i+2),
                                    kernel_initializer = init_ker)(l1)
    l2 = tf.keras.layers.Dense(2,
                                kernel_initializer = init_ker,dtype=dtype)(l1)
    jump_out = proto_jump_layer()([l2,xvals,rvals])
    
    
    ##Combines both networks and defines other NNs that are used for explainability. 
    u_out = sing_out+jump_out
    
    uh2 = tf.keras.Model(inputs=xvals,outputs=jump_out)
    
    u_model = tf.keras.Model(inputs=xvals,outputs = u_out)
    
    
    singular_model = tf.keras.Model(inputs=xvals,outputs=sing_out)
    singular_explain = tf.keras.Model(inputs=xvals,outputs=p_sing)
    
    u_model.summary()
    
    return u_model, uh2, singular_model, singular_explain


##Classical architecture. 
def make_model_naive(neurons,activation="tanh"):
    xvals = tf.keras.layers.Input(shape=(2,),dtype=dtype)
    l1 = tf.keras.layers.Dense(neurons,activation=activation,dtype=dtype)(xvals)
    for i in range(2):
        l1 = tf.keras.layers.Dense(neurons,activation=activation,dtype=dtype,name="l"+str(i+2))(l1)
    l2 = tf.keras.layers.Dense(1,dtype=dtype)(l1)
    
    u_model = tf.keras.Model(inputs=xvals,outputs=l2)
    u_model.summary()
    return u_model