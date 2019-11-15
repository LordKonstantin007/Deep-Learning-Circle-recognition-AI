# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:04:41 2019

@author: Cyrus
"""

import numpy as np

from keras.preprocessing import image
from keras.models import load_model

model = load_model('Model8.h5')


img_pred = image.load_img('./Test/test3.png', target_size = (200, 200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)



rslt = model.predict(img_pred)
print (rslt)
if rslt[0][0] == 1:
     prediction = "Viereck"
else:
     prediction = "Kreis"    
print (prediction) 

  


  
