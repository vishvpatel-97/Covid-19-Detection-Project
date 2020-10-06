#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow 
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image  
from tensorflow.keras.models import *
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


covid = load_model('covid.h5')


# In[3]:


def output(model,img_path,size):
    img_path=img_path
    img = image.load_img(img_path,target_size=size)
    img_arr = image.img_to_array(img,dtype='double')
    img_arr=img_arr/255
    img_arr=np.expand_dims(img_arr,axis=0)
    result=model.predict_classes(img_arr)
    
    if result==1:
        print("Positive")
    elif result==0:
        print("Negative")


# In[4]:


output(covid,'download.jpg',(224,224))


# In[5]:


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img,dtype='double')
    
    x=x/255
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)

    preds = model.predict_classes(x)
    
    if preds==0:
        print("Negative")
    elif preds==1:
        print("Positive")


# In[6]:


img_path='download.jpg'
model=load_model('covid.h5')
model_predict(img_path,model)


# In[ ]:




