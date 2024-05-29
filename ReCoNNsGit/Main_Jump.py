# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:12:51 2023

@author: jamie.taylor
"""


import tensorflow as tf
import numpy as np
from SRC.Postprocessing import post_process_H1
from SRC.Architecture_Jump import make_jump_model
from SRC.Loss_Jump import make_loss_jump, sol_exact, sigma
import matplotlib.pyplot as plt
import os

dtype="float64"


tf.keras.backend.set_floatx(
    dtype
)


##Define the square of the weights in the loss (PDE, jumps, BC and singular BC)
weights = tf.constant([1.,100.,10.],dtype=dtype)


###Create u_NN and the explainable components of the network
u_model, jump_explain_model = make_jump_model(30)


##Define the loss function
loss_model = make_loss_jump(u_model,jump_explain_model,1000)


###Define the loss and metrics used during training
def train_loss(y_pred,y_true):
    mid_loss =  tf.reduce_sum(tf.einsum("ij,i",y_true,weights)**0.5)
    return mid_loss**0.5

def metric_i(i):
    def f(y_pred,y_true):
        return y_true[i]
    f.__name__ = "metric_"+str(i)
    return f


iterations=50000



##Exponential factor for the learning rate during training
lr_fact = (10**-3)**(2/(iterations))

##Define the Learning rate scheduler
def scheduler(epoch, lr):
    if epoch< iterations/2:
        ans = lr
    else:
        ans = lr*lr_fact
    return ans

lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)

metrics = [metric_i(i) for i in range(3)]

optimizer = tf.keras.optimizers.Adam(learning_rate=10**-3)

loss_model.compile(optimizer=optimizer,loss=train_loss, metrics=metrics)


###Training
history1 = loss_model.fit(
    x = tf.constant([1.]),
    y = tf.constant([1.]),
    epochs=iterations,
    batch_size=1,
    validation_data = (tf.constant([1.]),tf.constant([1.])),
    callbacks = [tf.keras.callbacks.TerminateOnNaN(),lrs]
    )




####################
###Extract plots to show solutions. 
losstab =np.array(history1.history["loss"])
plt.plot(losstab)
plt.xscale("log")
plt.yscale("log")
plt.show()

for i in range(3):
    s = np.array(history1.history["metric_"+str(i)])
    plt.plot(s)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    
    
post_process_H1(u_model,sol_exact,500,500,maxval=1.,minval=-1.,change=True) 

 