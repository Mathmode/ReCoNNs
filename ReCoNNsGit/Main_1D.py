# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:41:02 2023

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from SRC.Loss_1D import make_loss_jump, make_loss_MC, sol_exact
from SRC.Architecture_1D import make_classic_model, make_jump_model


dtype="float64"


tf.keras.backend.set_floatx(
    dtype
)

neurons= 20
n=2500
iterations=5000


##Choice of architecture, either "reconn" or the name of an activation function
architecture = "relu"

##Choice of loss, or "MC" (any architecture) or "PINNs" (reconn only)
loss_fn = "MC"



###Create u_NN
if architecture == "reconn":
    u_model,jump_explain_model = make_jump_model(neurons)
else:
    u_model = make_classic_model(neurons,activation=architecture)
    
###Create the loss function, appropriate metrics for the problem
if loss_fn == "PINNs":
    loss_model = make_loss_jump(u_model, jump_explain_model, n)
    
    ##Define loss function 
    def train_loss(y_pred,y_true):
        return tf.reduce_sum(y_true**0.5)
    
    ##Define metrics to extract each component of the loss
    def metric_i(i):
        def f(y_pred,y_true):
            return y_true[i]
        f.__name__ = "metric_"+str(i)
        return f

    metrics= [metric_i(0),metric_i(1),metric_i(2)]
    
elif loss_fn == "MC":
    loss_model = make_loss_MC(u_model,n)
    
    metrics  = []


    def train_loss(y_pred,y_true):
        mid_loss = tf.reduce_sum(y_true)
        return mid_loss**0.5


##Training 
optimizer = tf.keras.optimizers.Adam(learning_rate=10**-3)

loss_model.compile(optimizer=optimizer,loss=train_loss,metrics=metrics)


history = loss_model.fit(
    x = tf.constant([1.]),
    y = tf.constant([1.]),
    epochs=iterations,
    batch_size=1,
    validation_data = (tf.constant([1.]),tf.constant([1.])),
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    )


###Extract properties of the solution
xtest = tf.constant([np.pi*(i+0.5)/300 for i in range(300)],dtype=dtype)

with tf.GradientTape(persistent=True) as t1:
    t1.watch(xtest)
    u = tf.squeeze(u_model(xtest))
    ue = sol_exact(xtest)
du = t1.gradient(u,xtest)
due = t1.gradient(ue,xtest)

plt.plot(xtest,u)
plt.plot(xtest,ue)
plt.show()

plt.plot(xtest,u-ue)
plt.show()

plt.plot(xtest,du)
plt.plot(xtest,due)
plt.show()

plt.plot(xtest,du-due)
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xscale("log")
plt.yscale("log")
plt.show()

h1_err = (tf.reduce_mean((due-du)**2)/tf.reduce_mean(due**2))**0.5
l2_err = (tf.reduce_mean((ue-u)**2)/tf.reduce_mean(u**2))**0.5

print("H1 err = ", 100*float(h1_err))
print("L2 err =", 100*float(l2_err))


