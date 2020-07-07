#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[2]:


IMAGE_SIZE = [224,224]

train_path = 'images/Train'
valid_path = 'images/Test'


# In[5]:


resnet = ResNet50(input_shape = IMAGE_SIZE +[3], weights = 'imagenet', include_top = False)


# In[6]:


resnet.summary()


# In[15]:


for layer in resnet.layers:
    layer.trainable = False


# In[16]:


folders = glob('images/Train/*')


# In[17]:


folders


# In[18]:


x = Flatten()(resnet.output)


# In[21]:


prediction = Dense(len(folders), activation = 'softmax')(x)

model = Model(inputs = resnet.input, outputs = prediction)


# In[22]:


model.summary()


# In[24]:


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[27]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[29]:


training_set = train_datagen.flow_from_directory('images/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[31]:


test_set = test_datagen.flow_from_directory('images/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[ ]:


r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

