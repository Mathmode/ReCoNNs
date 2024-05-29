# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:35:17 2023

@author: jamie.taylor
"""


import tensorflow as tf
from SRC.Architecture_4_materials import lamfun
import numpy as np

dtype="float64"



###Used to measure singular exponent during training
class measure_exponent(tf.keras.callbacks.Callback):
    def __init__(self,u_model,ind=-4):
        super(measure_exponent, self).__init__()
        self.lamlist=[]
        self.u_model=u_model
        self.ind=ind
    
    def on_epoch_begin(self, epoch, logs=None):
        
        ## DFR aug
        lamnew = lamfun(self.u_model.layers[self.ind].lam[0])
        
        tf.print(" lam =",lamnew)
        self.lamlist+=[float(lamnew)]