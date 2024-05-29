# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:41:24 2023

@author: jamie.taylor
"""


import tensorflow as tf
import numpy as np
from SRC.Postprocessing import post_process_H1

from SRC.Loss_L import make_loss_L, sol_exact, singular_sol
from SRC.Architecture_L import make_L_model, make_model_naive

from SRC.Architecture_4_materials import lamfun

from SRC.Callbacks import measure_exponent



import matplotlib.pyplot as plt

dtype="float64" 
tf.keras.backend.set_floatx(
    dtype
)

l0=2./3.


###Choice of u_NN - Either ReCoNN or classical architecture. 
use_reconn = True


if use_reconn:
    
    ##Create u_NN, explainable components and the loss model.
    u_model, uh2, singular_model, singular_explain = make_L_model(30,15)
    
    loss_model = make_loss_L(u_model,1000,singular_explain=singular_explain)
    
    
    ##Square of weights in the loss (PDE, BC, BC of angular function)
    weights = tf.constant([1,100.,1],dtype=dtype)
    
    
    ##Callback for measuring the singular exponent during training
    lams = measure_exponent(u_model,-5)
    
    callbacks = [tf.keras.callbacks.TerminateOnNaN(),lams]
    
else:
    
    ##Classical NN and loss model.
    u_model = make_model_naive(35)
    
    
    loss_model = make_loss_L(u_model,1000,singular_explain=False)
    
    
    ##Weights in the loss (PDE, BC)
    weights = tf.constant([1.,1000.],dtype=dtype)
    
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]


##Total number of iterations
iterations = 50000


##Define the learning rate scheduler
lr_fact = (10**-3)**(2/(iterations))

def scheduler(epoch, lr):
    if epoch< iterations/2:
        ans = lr
    else:
        ans = lr*lr_fact
    return ans

lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)

callbacks+=[lrs]


##Definition of the loss used in model.fit and metrics during training
def train_loss(y_pred,y_true):
    mid_loss = tf.reduce_sum(tf.einsum("ij,i",y_true,weights)**0.5)
    return mid_loss

def metric_i(i):
    def f(y_pred,y_true):
        return y_true[i]
    f.__name__ = "metric_"+str(i)
    return f



metrics = [metric_i(i) for i in range(len(loss_model(tf.constant([1.],dtype=dtype))))]

optimizer = tf.keras.optimizers.Adam(learning_rate=10**-3)

loss_model.compile(optimizer=optimizer,loss=train_loss, metrics=metrics)

###u_NN training. 
one=tf.constant([1],dtype=dtype)
history1 = loss_model.fit(
    x =one,
    y = one,
    epochs=iterations,
    batch_size=1,
    validation_data = (one,one),
    callbacks = callbacks
    )





#################
###Extract losses and metrics for illustration
losstab =np.array(history1.history["loss"])



plt.plot(losstab)
plt.xscale("log")
plt.yscale("log")
plt.show()

for i in range(len(metrics)):
    s = np.array(history1.history["metric_"+str(i)])
    plt.plot(s)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    
    
###Cutoff function used in the plots
def cut(x,y):
    jx = (tf.math.sign(x)+1)/2
    jy = (tf.math.sign(y)+1)/2
    return 1-(1-jx)*(1-jy)


###Plot of solution, errors, etc.
post_process_H1(u_model,sol_exact,500,500,maxval=1.,minval=-1.,cut=cut,change=True) 



##Plots for the angular function 
ts = tf.constant([-np.pi/2+i*np.pi*3/2/1000 for i in range(1000)],dtype="float64")


obtained_exponent= lamfun(u_model.layers[-5].lam)

delta_test=0.025
x_test = tf.stack([tf.math.cos(ts),tf.math.sin(ts)],axis=-1)


sing= singular_model(delta_test*x_test)/(delta_test**obtained_exponent)

plt.plot(ts,sing)
plt.plot(ts,singular_sol(tf.math.cos(ts),tf.math.sin(ts)))
plt.show()

with tf.GradientTape() as t1:
    t1.watch(ts)
    x_test = delta_test*tf.stack([tf.math.cos(ts),tf.math.sin(ts)],axis=-1)
    phi=singular_model(x_test)/(delta_test**obtained_exponent)
dphi=t1.gradient(phi,ts)
plt.scatter(ts,dphi)
plt.show()


###Comparison of exponent during training with exact value. 
plt.plot(np.log(np.array([i+1 for i in range(len(lams.lamlist))])),lams.lamlist)
plt.plot(np.log(np.array([i+1 for i in range(len(lams.lamlist))])),[2/3]*len(lams.lamlist))
plt.show()
