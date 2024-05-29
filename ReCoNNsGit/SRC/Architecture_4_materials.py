# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 10:49:18 2023

@author: jamie.taylor
"""


import tensorflow as tf
import numpy as np

dtype="float64"

sin=tf.math.sin
cos=tf.math.cos
pi=np.pi
Pi=np.pi


###delta1 and delta2 corresponding to the smooth cutoff function: eta(r)=1
### for r<delta1 and eta(r)=0 for r>delta2
delta1=tf.constant(0.5,dtype=dtype)
delta2=tf.constant(0.9,dtype=dtype)



def heaviside(x):
    return tf.math.maximum(tf.math.sign(x),0.)


###Unscaled version of eta equal to 0 for r<0 and 1 for r>1
def relcut(x):
    proto = -6*x**5 + 15*x**4 - 10*x**3 + 1
    plus = heaviside(x)
    mas = heaviside(1-x)
    return (1-plus)+plus*mas*proto

###Can be defined as a non-zero function to employ homogeneous Dirichlet BCs
def bc_cutoff(x,y):
    return tf.ones_like(x)


###Layer mapping X to X/||X|| to make outputs depend only on the angle. 
class angle_layer(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(angle_layer,self).__init__()
        
    
    def call(self,inputs):
        xy = inputs
        x,y=tf.unstack(xy,axis=-1)
        ang=tf.einsum("ij,i->ij",tf.stack([x,y],axis=-1),tf.math.sqrt(x**2+y**2)**-1)
        return ang


##Layer mapping X to ||X|| for dependence only on the radius
class r_layer(tf.keras.layers.Layer):    
    def __init__(self,**kwargs):
        super(r_layer,self).__init__()
    
    def call(self,inputs):
        x,y=tf.unstack(inputs,axis=-1)
        r = tf.sqrt(x**2+y**2)
        return r



###Function used to enforce jumps in the derivative, can be adjusted
def lelu(x):
    return tf.math.abs(x) 


##Mapping applied to lambda to give distinct dependence, can be adjusted, 
##taken as identity for simplicity. 
def lamfun(s):
    return s


####Penultimate layer of the singular component. Returns distinct tensors 
###used to construct the singular component to allow explainability
###and incorporation into the loss function. 
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
        p1,px,py = tf.unstack(phi,axis=-1)
        radcut = relcut(self.a*r+self.b)        
        
        return p1,px,py,radcut,lamfun(self.lam[0]),r

###Last layer of the singular component of u_NN, assembles various tensors. 
class  singular_out_layer(tf.keras.layers.Layer):
    def __init__(self,):
        super(singular_out_layer,self).__init__()
    
    def call(self,inputs):
        z,xy = inputs
        p1,px,py,radcut,lam,r =z
        
        x,y=tf.unstack(xy,axis=-1)
        
        u = (p1+px*lelu(x)+py*lelu(y))*radcut*(r**lam)
        return u


###Penultimate layer of the cartesian component of u_NN.
###Outputs components independently to allow explainability. 
###Introduces discontinuities in the gadient. 
class proto_jump_layer(tf.keras.layers.Layer):
    def __init__(self,):
        super(proto_jump_layer,self).__init__()
        
        ###cutoff values for r in the cutoff function. 
        self.delta1=delta1
        self.delta2=delta2
        
        ##scale factors for eta. 
        self.a = -1/(self.delta1 - self.delta2)
        self.b = self.delta1/(self.delta1 - self.delta2)
    
    def call(self,inputs):
        
        phi,xy,r = inputs
        p11,p12,px1,px2,py1,py2 = tf.unstack(phi,axis=-1)
        
        cut = relcut(self.a*r+self.b)
        
        p1 = p11+cut*p12
        px = px1+cut*px2
        py = py1+cut*py2
        
        
        x,y = tf.unstack(xy,axis=-1)
        return p1,px,py,x,y,x
    
    
###Output layer for the Cartesian part of u_NN. Assembles the components
###from the previous layer. 
class jump_out_layer(tf.keras.layers.Layer):
    def __init__(self,):
        super(jump_out_layer,self).__init__()
    def call(self,inputs):
        
        p1,px,py,x,y,dummy=inputs
        ans= p1+px*lelu(x)+py*lelu(y)

        return ans


    
###Create the full ReCoNN for the 4-materials problem. 
###Effectively uses two parallel networks, one in polar coordinates and the other
###in Cartesian coordinates, 
def make_4_materials_model(neurons,neurons2,activation="tanh"):
    
    init_ker = tf.keras.initializers.GlorotUniform()
    
    
    ##Input
    xvals = tf.keras.layers.Input(shape=(2,), name="x_input",dtype=dtype)
    
    ##Mapping to polar coordinates
    rvals = r_layer()(xvals)
    angvals = angle_layer()(xvals)
    
    
    
    ####Creates the angular/singular component of the ReCoNN. 
    tl1 = tf.keras.layers.Dense(neurons2,activation=activation,dtype=dtype,
                                kernel_initializer = init_ker)(angvals)
    for i in range(2):
        tl1 = tf.keras.layers.Dense(neurons2,activation=activation,dtype=dtype,
                                    kernel_initializer = init_ker)(tl1)

    tl4 = tf.keras.layers.Dense(3,dtype=dtype,
                                kernel_initializer = init_ker)(tl1)
    p_sing = proto_singular_layer()([rvals,tl4,angvals])
    sing_out = singular_out_layer()([p_sing,angvals])
    
    
    ###Creates the Cartesian component of the ReCoNN. 
    l1 = tf.keras.layers.Dense(neurons,activation=activation,dtype=dtype,
                                kernel_initializer = init_ker)(xvals)
    for i in range(2):
        l1 = tf.keras.layers.Dense(neurons,activation=activation,dtype=dtype,name="l"+str(i+2),
                                    kernel_initializer = init_ker)(l1)
    l2 = tf.keras.layers.Dense(6,
                                kernel_initializer = init_ker,dtype=dtype)(l1)
    p_jump_out = proto_jump_layer()([l2,xvals,rvals])
    jump_out = jump_out_layer()(p_jump_out)
    
    
    
    ##Total output
    u_out = sing_out+jump_out
    
    
    
    ##Smooth part of u
    uh2 = tf.keras.Model(inputs=xvals,outputs=jump_out)
    ##Model used to describe jumps at the interfaces. 
    jump_explain = tf.keras.Model(inputs=xvals,outputs =p_jump_out)
    
    
    ###Total model u_NN
    u_model = tf.keras.Model(inputs=xvals,outputs = u_out)
    
    ###Only the singular part of u_NN
    singular_model = tf.keras.Model(inputs=xvals,outputs=sing_out)
    ###Quantities needed to explain the jump in the derivative in the 
    ###angular function. 
    singular_explain = tf.keras.Model(inputs=xvals,outputs=p_sing)
    
    u_model.summary()
    
    return u_model, uh2, jump_explain, singular_model, singular_explain

###Classical Fully-Connected Feed-Forward NN for comparison against ReCoNN. 
def make_model_naive(neurons,activation="tanh"):
    xvals = tf.keras.layers.Input(shape=(2,),dtype=dtype)
    l1 = tf.keras.layers.Dense(neurons,activation=activation,dtype=dtype)(xvals)
    for i in range(2):
        l1 = tf.keras.layers.Dense(neurons,activation=activation,dtype=dtype,name="l"+str(i+2))(l1)
    l2 = tf.keras.layers.Dense(1,dtype=dtype)(l1)
    
    u_model = tf.keras.Model(inputs=xvals,outputs=l2)
    u_model.summary()
    return u_model
    
    
    