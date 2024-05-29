# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:23:05 2023

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np

from SRC.Architecture_4_materials import  relcut, heaviside, bc_cutoff, lelu

delta1=0.025

dtype="float64"

sin=tf.math.sin
cos=tf.math.cos
pi=np.pi
Pi=np.pi


###We define the parameters corresponding to the exact solution

## Sigmas 1,2,3,4
s1=1
s2=2
s3=3
s4=4
a1 = 3.58396766000856
a2 = 3.28530926421398
a3 = 2.47465193074208
a4 = 2.11503551932097
b1 = -2.00351054735044
b2 = -0.667838639957490
b3 = -1.04946191312213
b4 = -0.586064428952415
l0=0.8599513039


###These are the exponents that appear in the weighting of the loss
###for the interfacial and PDE components. 
exponent_interfaces = 1
exponent_bulk= 2




###Used to define the weighting function omega for the PDE and interface losses
def relcut_C0(x):
    return tf.nn.relu(x)-tf.nn.relu(x-1)


##We define sigma as a piecewise function
def sigma(xy):
    x,y = tf.unstack(xy,axis=-1)
    sx = heaviside(x)
    sy = heaviside(y)
    return s1*sx*sy + s2*(1-sx)*sy + s3*(1-sx)*(1-sy) + s4*sx*(1-sy)

###We define the function taking x,y to the angle with the x-axis, measured from
###0 to 2pi
def th(x,y):
    pth = tf.math.atan2(y,x)
    
    return pth + 2*np.pi*(tf.math.sign(-pth)+1)/2


###Definition of the singular component of the solution. 
def singular_sol(xy):
    x,y = tf.unstack(xy,axis=-1)
    t= th(x,y)
    r = (x**2+y**2)**0.5
    sx = (tf.sign(x)+1)/2
    sy = (tf.sign(y)+1)/2
    
    sol1 = a1*sin(l0*t)+b1*cos(l0*t)
    sol2 = a2*sin(l0*t)+b2*cos(l0*t)
    sol3 = a3*sin(l0*t)+b3*cos(l0*t)
    sol4 = a4*sin(l0*t)+b4*cos(l0*t)
    
    return (sol1*sx*sy+sol2*(1-sx)*sy+sol3*(1-sx)*(1-sy)+sol4*sx*(1-sy))*(r**l0)


###The exact solution. 
def sol_exact(xy):
    x,y = tf.unstack(xy,axis=-1)
    sing = singular_sol(xy)
    return cos(x*pi/2)*cos(y*pi/2)*sing


###The right-hand side term of sigma(x,y)Delta u(x,y)=f(x,y),
###Defined in terms of the exact solution. 
def f_rhs(xy):
    x,y = tf.unstack(xy,axis=-1)
    with tf.GradientTape(persistent=True) as t1:
        t1.watch(x)
        t1.watch(y)
        solex = singular_sol(tf.stack([x,y],axis=-1))
    dsolx = t1.gradient(solex,x)
    dsoly = t1.gradient(solex,y)
    
    return sigma(xy)*(-solex*Pi**2*cos(Pi*x/2)*cos(y*Pi/2)/2 - dsolx*Pi*sin(Pi*x/2)*cos(y*Pi/2) - Pi*sin(y*Pi/2)*cos(Pi*x/2)*dsoly)



###Monte Carlo sampling strategy, strategic over each quadrant. 
def quad_sample(n):
    x = tf.concat([tf.random.uniform([n],dtype=dtype,minval=0,maxval=1),
                  tf.random.uniform([n],dtype=dtype,minval=0,maxval=1),
                  tf.random.uniform([n],dtype=dtype,minval=-1,maxval=0),
                  tf.random.uniform([n],dtype=dtype,minval=-1,maxval=0)],axis=-1)
    y = tf.concat([tf.random.uniform([n],dtype=dtype,minval=0,maxval=1),
                  tf.random.uniform([n],dtype=dtype,minval=-1,maxval=0),
                  tf.random.uniform([n],dtype=dtype,minval=-1,maxval=0),
                  tf.random.uniform([n],dtype=dtype,minval=0,maxval=1)],axis=-1)
    return x,y



###Definition of the component of the loss to enforce the Dirichlet BC
class loss_Dirichlet(tf.keras.layers.Layer):
    def __init__(self,u_model,n):
        super(loss_Dirichlet,self).__init__()
        self.u_model = u_model
        self.n=int(n/4)
        
        
    
    def call(self,inputs):
        
        
        ###Sampling over the boundary
        pvals = tf.random.uniform([self.n],minval=-1,maxval=1,dtype=dtype)
        x1 = tf.stack([tf.ones_like(pvals),pvals],axis=-1)
        
        pvals = tf.random.uniform([self.n],minval=-1,maxval=1,dtype=dtype)
        xm1 = tf.stack([-tf.ones_like(pvals),pvals],axis=-1)
        
        pvals = tf.random.uniform([self.n],minval=-1,maxval=1,dtype=dtype)
        y1 = tf.stack([pvals,tf.ones_like(pvals)],axis=-1)
        
        pvals = tf.random.uniform([self.n],minval=-1,maxval=1,dtype=dtype)
        ym1 = tf.stack([pvals,-tf.ones_like(pvals)],axis=-1)
        xy = tf.concat([x1,xm1,y1,ym1],axis=0)
        
        ###Evaluation of u and the loss at sampled points
        u = self.u_model(xy)
        
        ubc = u**2
        
        bc_loss = tf.reduce_mean(ubc)*8
        
        return bc_loss



###The component of the loss corresponding to the jump in the flux over
###the interface described by x=0. 
def full_jump_x(singular_explain,jump_explain,sleft,sright,n,sign):
    ##Sampling
    y0 = tf.random.uniform([n],dtype=dtype)*sign
    x0 = 0*y0
    
    ##Evaluation of the terms necessary to calculate the flux 
    ##on each side of the interface
    with tf.GradientTape() as t1:
        t1.watch(x0)
        xy = tf.stack([x0,y0],axis=-1)
        bc_cut = bc_cutoff(x0,y0)
        jump_p1,jump_px,jump_py,rc,l,r = jump_explain(xy)
        sing_p1,sing_px,sing_py,rc,l,r = singular_explain(xy)
        smooth_sing = (sing_p1+sing_py*lelu(y0/r))*rc*(r**l)
        smooth_jump = bc_cut*(jump_p1+jump_py*lelu(y0))
        phi1 = smooth_sing+smooth_jump
    dphi1 = t1.gradient(phi1,x0)
    
    sing_fact = rc*r**(l-1)
    
    
    ##Evaluation of left- and right-limits. 
    dleft = dphi1 -(jump_px*bc_cut+sing_px*sing_fact)
    dright = dphi1 + (jump_px*bc_cut+sing_px*sing_fact)
    
    
    ##Weighting function in the loss
    omega = relcut_C0(tf.math.abs(y0/delta1))**exponent_interfaces
    
    jump = (sleft*dleft-sright*dright)**2*omega  
    
    return tf.reduce_mean(jump)



###The component of the loss corresponding to the jump in the flux over
###the interface described by y=0. 
def full_jump_y(singular_explain,jump_explain,stop,sbottom,n,sign):
    ##Sampling
    x0 = tf.random.uniform([n],dtype=dtype)*sign
    y0 = 0*x0
    
    ##Evaluation of the terms necessary to calculate the flux 
    ##on each side of the interface
    with tf.GradientTape() as t1:
        t1.watch(y0)
        xy = tf.stack([x0,y0],axis=-1)
        bc_cut = bc_cutoff(x0,y0)
        jump_p1,jump_px,jump_py,rc,l,r = jump_explain(xy)
        sing_p1,sing_px,sing_py,rc,l,r = singular_explain(xy)
        smooth_sing = (sing_p1+sing_px*lelu(x0/r))*rc*(r**l)
        smooth_jump = bc_cut*(jump_p1+jump_px*lelu(x0))
        phi1 = smooth_sing+smooth_jump
    dphi1 = t1.gradient(phi1,y0)
    
    sing_fact = rc*(r**(l-1))
    
    ##Weighting function in the loss
    omega = relcut_C0(tf.math.abs(x0/delta1))**exponent_interfaces

    ##Evaluation of left- and right-limits. 
    dbottom = dphi1 -(jump_py*bc_cut+sing_py*sing_fact)
    dtop = dphi1+(jump_py*bc_cut+sing_py*sing_fact)
    
    
    jump = (sbottom*dbottom-stop*dtop)**2*omega
    return tf.reduce_mean(jump)


###Total component of the loss corresponding to the continuity of the flux
###across interfaces
class jump_loss_full(tf.keras.layers.Layer):
    def __init__(self,singular_explain,jump_explain,n):
        super(jump_loss_full,self).__init__()
        self.singular_explain=singular_explain
        self.jump_explain=jump_explain
        self.n=int(n/4)
    
    def call(self,inputs):
        one = tf.constant(1,dtype=dtype)
        
        jump_y_1 = full_jump_y(self.singular_explain,self.jump_explain,s1,s4,self.n,one)
        jump_y_m1 = full_jump_y(self.singular_explain,self.jump_explain,s2,s3,self.n,-one)
        jump_x_1 = full_jump_x(self.singular_explain,self.jump_explain,s2,s1,self.n,one)
        jump_x_m1 = full_jump_x(self.singular_explain,self.jump_explain,s3,s4,self.n,-one)
        ans = jump_y_1+jump_y_m1+jump_x_1+jump_x_m1
        return ans


###Evaluation of the component of the no-flux condition of the angular
###solution at the points corresponding to x=0
def jump_x_SL(singular_explain,sleft,sright,sign):
    y0 = tf.constant([delta1/2],dtype=dtype)*sign
    x0 = 0*y0
    
    with tf.GradientTape() as t1:
        t1.watch(x0)
        xy = tf.stack([x0,y0],axis=-1)
        sing_p1,sing_px,sing_py,rc,l,r = singular_explain(xy)
        smooth_sing = (sing_p1+sing_py*lelu(y0/r))*rc*(r**l)
        phi1 = smooth_sing
    dphi1 = t1.gradient(phi1,x0)
    
    sing_fact = rc*r**(l-1)
    
    dleft = dphi1 -(sing_px*sing_fact)
    dright = dphi1 + (sing_px*sing_fact)
    
    jump = (sleft*dleft-sright*dright)  
    
    return tf.reduce_mean(jump**2)


###Evaluation of the component of the no-flux condition of the angular
###solution at the points corresponding to y=0
def jump_y_SL(singular_explain,stop,sbottom,sign):
    x0 = tf.constant([delta1/2],dtype=dtype)*sign
    y0 = 0*x0
    with tf.GradientTape() as t1:
        t1.watch(y0)
        xy = tf.stack([x0,y0],axis=-1)
        sing_p1,sing_px,sing_py,rc,l,r = singular_explain(xy)
        smooth_sing = (sing_p1+sing_px*lelu(x0/r))*rc*(r**l)
        phi1 = smooth_sing
    dphi1 = t1.gradient(phi1,y0)
    
    sing_fact = rc*(r**(l-1))
    
    dbottom = dphi1 -(sing_py*sing_fact)
    dtop = dphi1+(sing_py*sing_fact)
    
    
    jump = (sbottom*dbottom-stop*dtop)
    return tf.reduce_mean(jump**2)


###Layer to evaluate the loss that corresponds to the continuity of the flux
###of the angular function. 
class jump_loss_SL(tf.keras.layers.Layer):
    def __init__(self,singular_explain):
        super(jump_loss_SL,self).__init__()
        self.singular_explain=singular_explain
    
    def call(self,inputs):
        one = tf.constant(1,dtype=dtype)
        
        jump_y_1 = jump_y_SL(self.singular_explain,s1,s4,one)
        jump_y_m1 = jump_y_SL(self.singular_explain,s2,s3,-one)
        jump_x_1 = jump_x_SL(self.singular_explain,s2,s1,one)
        jump_x_m1 = jump_x_SL(self.singular_explain,s3,s4,-one)
        ans = jump_y_1+jump_y_m1+jump_x_1+jump_x_m1
        return ans


###Layer to evaluate the PDE component of the loss. 
class loss_PINNs_4_materials(tf.keras.layers.Layer):
    def __init__(self,u_model,n):
        super(loss_PINNs_4_materials,self).__init__()
        self.u_model=u_model
        self.n= int(n/4)
        
        self.delta=delta1
        
    def call(self,inputs):
        
        
        ##Sampling
        x,y = quad_sample(self.n)
        
        
        ##Evaluation of all necessary derivatives.
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            t1.watch(y)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x)
                t2.watch(y)
                xy = tf.stack([x,y],axis=-1)
                u = self.u_model(xy)
            dux,duy = t2.gradient(u,[x,y])
            del t2
        duxx = t1.gradient(dux,x)
        duyy = t1.gradient(duy,y)
        del t1
        
        ##Evaluation of the weighting function
        r=tf.sqrt(x**2+y**2)
        
        omega=relcut_C0(r/self.delta)**exponent_bulk
        
        
        ##Evaluation of the residual
        sig = sigma(xy)
        
        f_xy=f_rhs(xy)
        
        residual = (sig*(duxx+duyy)-f_xy)
        
        p_loss = tf.reduce_mean(residual**2*omega)*4
        
        return p_loss


###used to combine components of the loss into a single tensor
class stack_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(stack_layer,self).__init__()
        
    def call(self,inputs):
        return tf.stack([inputs],axis=-1)

###used to create the loss model with each component. 
def make_loss_PINNs_4(u_model,uh2,singular_model,singular_explain,jump_explain,n):
    xvals = tf.keras.Input(shape=(1,),dtype=dtype)   
    
    pinns_loss = loss_PINNs_4_materials(u_model,n)(xvals)
        
    bc_loss = loss_Dirichlet(u_model,n)(xvals)
        
    jump_loss=jump_loss_full(singular_explain,jump_explain,n)(xvals)
    
    SL_jump = jump_loss_SL(singular_explain)(xvals)
    
    out = stack_layer()([pinns_loss,jump_loss,bc_loss,SL_jump])
    
    loss_model = tf.keras.Model(inputs=xvals,outputs=out)
    
    return loss_model



###Loss function used to approximate the exact solution using the 
###H1 norm discretised via MC scheme. 
class loss_MC(tf.keras.layers.Layer):
    def __init__(self,u_model,u_exact,n):
        super(loss_MC,self).__init__()
        self.u_model=u_model
        self.n= int(n/4)
        self.u_exact = u_exact
        
    def call(self,inputs):
        
        ###Sampling 
        x,y = quad_sample(self.n)
        
        ##evaluation of the derivatives
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(x)
            t2.watch(y)
            xy = tf.stack([x,y],axis=-1)
            u = tf.squeeze(self.u_model(xy))-self.u_exact(xy)
        dux = t2.gradient(u,x)
        duy = t2.gradient(u,y)
        del t2
        
        ##Loss evaluation
        loss = tf.reduce_mean(u**2+dux**2+duy**2)
        
        return loss
    

###Creates the loss model for approximation in H1 norm. 
def make_MC_model(u_model,u_exact,n):
    xvals = tf.keras.Input(shape=(1,),dtype=dtype)
    output = loss_MC(u_model,u_exact,n)(xvals)
    loss_model = tf.keras.Model(inputs=xvals,outputs=output)
    return loss_model

