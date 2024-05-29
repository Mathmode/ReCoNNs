# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 10:46:34 2023

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from SRC.Architecture_4_materials import make_4_materials_model, lamfun,make_model_naive
from SRC.Loss_4_materials import make_loss_PINNs_4, sol_exact, singular_sol, sigma,l0,make_MC_model

from SRC.Postprocessing import post_process_H1



from SRC.Callbacks import measure_exponent



dtype = "float64"

tf.keras.backend.set_floatx(
    dtype
)



iterations=50

##Choice of loss function. Either PINNs if true, or MC if false. 
pinns_loss = False

##Choice of NN architecture. If false, classical architecture. Does not work 
##with PINNs
use_reconn = True

##Define the square of the weights in the loss (PDE, jumps, BC and singular BC)


if pinns_loss:
    
    ###Create u_NN and the explainable components of the network
    u_model, uh2, jump_explain, singular_model, singular_explain = make_4_materials_model(30,15)
    
    ##Define the loss model
    loss_model = make_loss_PINNs_4(u_model,uh2,singular_model,singular_explain,jump_explain,1000)
    
    
    ##square of weights corresponding to PDE, interface, Dirichlet BC, 
    ##interface of angular function
    weights = tf.constant([1.,10,100,1.],dtype=dtype)

    ###Define the loss and metrics used during training
    def train_loss(y_pred,y_true):
        mid_loss =  tf.reduce_sum(tf.einsum("ij,i",y_true,weights)**0.5)
        return mid_loss
    
    def metric_i(i):
        def f(y_pred,y_true):
            return y_true[i]
        f.__name__ = "metric_"+str(i)
        return f
        
    ##Define a list of metrics to be measured during training
    metrics = [metric_i(i) for i in range(len(loss_model(tf.constant([1.]))))]
else:
    if use_reconn:
        u_model, uh2, jump_explain, singular_model, singular_explain = make_4_materials_model(30,15)
        
        
        loss_model = make_MC_model(u_model,sol_exact,5000)

    else:
        u_model = make_model_naive(36)
        
        
        loss_model = make_MC_model(u_model,sol_exact,5000)
        
    metrics = []

        
    def train_loss(y_pred,y_true):
        mid_loss = y_true**0.5
        return mid_loss



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

callbacks = [lrs, tf.keras.callbacks.TerminateOnNaN()]

if use_reconn:
    lams = measure_exponent(u_model,-4)
    callbacks +=[lams]




##Define the optimiser
optimizer = tf.keras.optimizers.Adam(learning_rate=10**-3)

##Compile with optimiser
loss_model.compile(optimizer=optimizer,loss=train_loss, metrics=metrics)

###Train the NN
history1 = loss_model.fit(
    x = tf.constant([1.]),
    y = tf.constant([1.]),
    epochs=iterations,
    batch_size=1,
    validation_data = (tf.constant([1.]),tf.constant([1.])),
    callbacks = callbacks
    )

###We extract from the history metrics to be plotted
losstab =np.array(history1.history["loss"])
losstabv =np.array(history1.history["val_loss"])
plt.plot(losstab)
plt.plot(losstabv)
plt.xscale("log")

plt.yscale("log")
plt.show()


if pinns_loss:
    ##We plot each metric in turn
    for i in range(len(metrics)):
        s = np.array(history1.history["metric_"+str(i)])
        sv = np.array(history1.history["val_metric_"+str(i)])
        plt.plot((s))
        plt.plot((sv))
        plt.xscale("log")
        plt.yscale("log")
        plt.show()


###Plot the obtained solution
post_process_H1(u_model,sol_exact,500,500,maxval=1,minval=-1)

if use_reconn:
    ##Final value of the singular exponent. 
    obtained_exponent= lamfun(u_model.layers[-4].lam)
    
    ###We obtain the angular function and its flux for comparison
    ts = tf.constant([2*(i+0.5)*np.pi/1000 for i in range(1000)])
    delta_test=0.025
    with tf.GradientTape(persistent=True) as t1:
        t1.watch(ts)
        newx = delta_test*tf.stack([tf.math.cos(ts),tf.math.sin(ts)],axis=-1)
        phi=singular_model(newx)/(delta_test**obtained_exponent)
        
        sing = singular_sol(newx)/(delta_test**l0)
    dphi=t1.gradient(phi,ts)
    dsing = t1.gradient(sing,ts)
    plt.plot(ts,sigma(newx)*dphi)
    plt.plot(ts,sigma(newx)*dsing)
    plt.show()

    ###We plot the exponent during training against the true value
    plt.plot((np.array([i+1 for i in range(len(lams.lamlist))])),lams.lamlist)
    plt.plot((np.array([i+1 for i in range(len(lams.lamlist))])),[l0]*len(lams.lamlist))
    plt.xscale('log')
    plt.show()
