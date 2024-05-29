# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:34:00 2023

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from SRC.Loss_4_materials import sigma, f_rhs, relcut_C0




dtype="float64"

tf.keras.backend.set_floatx(
    dtype
)


def nocut(x,y):
    return tf.ones_like(x,dtype=dtype)



def post_process_H1(u_model,target,nxt,nyt,maxval=np.pi,minval=0.,cut=nocut,change=False):
        
    xtest = tf.constant([(i+0.5)/nxt for i in range(nxt)],dtype=dtype)*(maxval-minval)+minval
    ytest = tf.constant([(i+0.5)/nyt for i in range(nyt)],dtype=dtype)*(maxval-minval)+minval
    XT,YT = tf.meshgrid(xtest,ytest)
    xt = tf.reshape(XT,[nxt*nyt])
    yt = tf.reshape(YT,[nxt*nyt])
    
    

    with tf.GradientTape(persistent=True) as t1:
        t1.watch(xt)
        t1.watch(yt)
        xyt = tf.stack([xt,yt],axis=-1)
        ua = tf.squeeze(u_model(xyt))*cut(xt,yt)
        if change:
            ue = target(xt,yt)*cut(xt,yt)
        else:
            ue = target(xyt)*cut(xt,yt)
        er = ua-ue
    uax =t1.gradient(ua,xt)
    uay=t1.gradient(ua,yt)
    uex = t1.gradient(ue,xt)
    uey=t1.gradient(ue,yt)
    del t1
    
    
    

    plt.contourf(XT,YT,tf.reshape(ua,[nyt,nxt]))
    plt.colorbar()
    plt.title("u approx")
    plt.show()
    
    plt.contourf(XT,YT,tf.reshape(ue,[nyt,nxt]))
    plt.colorbar()
    plt.title("u exact")
    plt.show()
    
    
    
    plt.contourf(XT,YT,tf.reshape(((uax**2+uay**2))**0.5,[nyt,nxt]))
    plt.colorbar()
    plt.title("Grad approx")
    plt.show()
    
    
    plt.contourf(XT,YT,tf.reshape((uex**2+uey**2)**0.5,[nyt,nxt]))
    plt.colorbar()
    plt.title("Grad Exact")
    plt.show()
    
    
    plt.contourf(XT,YT,tf.reshape(tf.math.abs(er),[nyt,nxt]))
    plt.colorbar()
    plt.title("|u-u*|")
    plt.show()
    
    plt.contourf(XT,YT,tf.reshape(((uax-uex)**2+(uay-uey)**2)**0.5,[nyt,nxt]))
    plt.colorbar()
    plt.title("|Du-Du*|")
    plt.show()
    
    
    h1err = tf.reduce_mean(((uax-uex)**2+(uay-uey)**2))
    h1sol = tf.reduce_mean((uex**2+uey**2))
    print("h1 rel-er = ",float((h1err/h1sol)**0.5)*100)
    l2err = tf.reduce_mean(((ua-ue)**2))
    l2sol = tf.reduce_mean((ue**2))
    print("l2 rel-er = ",float((l2err/l2sol)**0.5)*100)
    