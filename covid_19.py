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


train_path = r'C:\Users\SANJAY\Desktop\Covid-19_Project\Covid19-dataset\train'
test_path = r'C:\Users\SANJAY\Desktop\Covid-19_Project\Covid19-dataset\test'


# In[3]:


model = Sequential()
model.add(Conv2D(32,kernel_size=(4,4),activation='relu', input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[4]:


model.summary()


# In[5]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_data = ImageDataGenerator(rescale = 1./255)


# In[6]:


train_set = train_data.flow_from_directory(train_path,target_size = (224,224),batch_size = 12,class_mode = 'binary')
test_set = test_data.flow_from_directory(test_path,target_size = (224,224),batch_size = 12,class_mode = 'binary')


# In[7]:


train_set.class_indices


# In[8]:


covid=model.fit(train_set,
                         steps_per_epoch = len(train_set),
                         epochs = 20,
                         validation_data = test_set,
                         validation_steps = len(test_set))


# In[9]:


plt.plot(covid.history['loss'], label='train loss')
plt.plot(covid.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(covid.history['accuracy'], label='train acc')
plt.plot(covid.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[10]:


get_ipython().system('pip install pyyaml h5py')


# In[11]:


model.save('covid.h5')


# In[ ]:




