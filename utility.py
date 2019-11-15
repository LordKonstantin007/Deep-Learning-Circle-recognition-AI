# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 18:16:56 2019

@author: Cyrus
"""

from keras.utils import plot_model
from keras.models import load_model

model = load_model('simple.h5')
plot_model(model, to_file='./model.png')