#!/usr/bin/env python
# coding: utf-8

# # Build a Transfer Learning image classification model using the VGG16 & VGG19 (pre-trained network).

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as layers


# # Prepare and Review Dataset

# In[2]:


# Test and Train path
train_path = "E://Users//kishore//Downloads//seg_train"
test_path = "E://Users//kishore//Downloads//seg_test"


# In[ ]:


pip install glob


# In[11]:


from glob import glob

# Rest of your code
train_path ="E://Users//kishore//Downloads//seg_train"
numberOfClass = len(glob(train_path + "/*"))
print("Number Of Class: ", numberOfClass)


# In[12]:


from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt

# Rest of your code
train_path = "C://Users//kishore//0.jpg"
img = load_img(train_path)
plt.imshow(img)
plt.axis("off")


# In[14]:


from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Rest of your code
train_path = "C://Users//kishore//0.jpg"
img = load_img(train_path)
image_array = img_to_array(img)
print(image_array.shape)


# In[16]:


from keras.preprocessing.image import ImageDataGenerator

# Rest of your code
train_path = "E://Users//kishore//Downloads//seg_train"
test_path =  "E://Users//kishore//Downloads//seg_test"

train_data = ImageDataGenerator().flow_from_directory(train_path, target_size=(224, 224))
test_data = ImageDataGenerator().flow_from_directory(test_path, target_size=(224, 224))


# # Visualization 

# In[ ]:


import os
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt

# Correct the train_path
train_path = "E://Users//kishore//Downloads//seg_train//seg_train//"

for i in os.listdir(train_path):
    for j in os.listdir(os.path.join(train_path, i)):  # Use os.path.join to handle path concatenation
        img_path = os.path.join(train_path, i, j)
        img = load_img(img_path)
        plt.imshow(img)
        plt.axis("off")
        plt.show()


# # VGG16 

# In[2]:


from keras.applications import VGG16

# Rest of your code
vgg16 = VGG16()


# In[3]:


# Layers of vgg16 
vgg16.summary()


# In[4]:


# layers of vgg16
vgg16_layer_list = vgg16.layers
for i in vgg16_layer_list:
    print(i)


# In[6]:


from keras.applications import VGG16
from keras.models import Sequential

# Rest of your code
vgg16 = VGG16()

# Create a Sequential model
vgg16Model = Sequential()

# Add the layers of VGG16 to your model
for layer in vgg16.layers[:-1]:  # Exclude the last layer (output layer) of VGG16
    vgg16Model.add(layer)


# In[7]:


# the final version of the model
vgg16Model.summary()


# In[8]:


# Close the layers of vgg16
for layers in vgg16Model.layers:
    layers.trainable = False


# In[11]:


from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense

# Assuming you have defined numberOfClass earlier in your code
numberOfClass = 10  # Replace with the actual number of classes

vgg16 = VGG16()
vgg16Model = Sequential()

for layer in vgg16.layers[:-1]:
    vgg16Model.add(layer)

vgg16Model.add(Dense(numberOfClass, activation="softmax"))


# In[12]:


# After I added last layer in created model.
vgg16Model.summary()


# In[13]:


# I create compile part.
vgg16Model.compile(loss = "categorical_crossentropy",
             optimizer = "rmsprop",
             metrics = ["accuracy"])


# # Training Model

# In[15]:


from keras.preprocessing.image import ImageDataGenerator

# Assuming you have defined train_path and test_path earlier in your code
train_path = "E://Users//kishore//Downloads//seg_train"
test_path ="E://Users//kishore//Downloads//seg_test"

# Using ImageDataGenerator for training data
train_data_gen = ImageDataGenerator(rescale=1./255)  # You can add more augmentations if needed
train_data = train_data_gen.flow_from_directory(train_path, target_size=(224, 224), batch_size=32, class_mode='categorical')


# In[16]:


# Save the weights of model
vgg16Model.save_weights("deneme.h5")


# In[19]:


# Assuming you have defined test_path earlier in your code
test_path = "E://Users//kishore//Downloads//seg_test"

# Using ImageDataGenerator for testing data
test_data_gen = ImageDataGenerator(rescale=1./255)  # You can add more augmentations if needed
test_data = test_data_gen.flow_from_directory(test_path, target_size=(224, 224), batch_size=32, class_mode='categorical')


# In[ ]:


from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense

# Assuming you have defined numberOfClass and train_data earlier in your code
numberOfClass = 10  # Replace with the actual number of classes
train_data = ...    # Define your train_data using ImageDataGenerator

# Build the VGG16 model
vgg16 = VGG16(include_top=False, input_shape=(224, 224, 3))  # Adjust input shape as needed

# Create a Sequential model and add the VGG16 layers
vgg16Model = Sequential()
for layer in vgg16.layers:
    vgg16Model.add(layer)

# Add your custom Dense layer
vgg16Model.add(Dense(numberOfClass, activation="softmax"))

# Compile the model (specify optimizer, loss, metrics as needed)
vgg16Model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
hist_vgg16 = vgg16Model.fit_generator(train_data, 
                                      steps_per_epoch=1600 // batch_size, 
                                      epochs=10, 
                                      validation_data=test_data, 
                                      validation_steps=800 // batch_size)


# In[ ]:


from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Assuming you have defined numberOfClass and train_path and test_path earlier in your code
numberOfClass = 10  # Replace with the actual number of classes
train_path = "E://Users//kishore//Downloads//seg_train"
test_path = "E://Users//kishore//Downloads//seg_test"

# Using ImageDataGenerator for training data
train_data_gen = ImageDataGenerator(rescale=1./255)  # You can add more augmentations if needed
train_data = train_data_gen.flow_from_directory(train_path, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Using ImageDataGenerator for testing data
test_data_gen = ImageDataGenerator(rescale=1./255)
test_data = test_data_gen.flow_from_directory(test_path, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Build the VGG16 model
vgg16 = VGG16(include_top=False, input_shape=(224, 224, 3))  # Adjust input shape as needed

# Create a Sequential model and add the VGG16 layers
vgg16Model = Sequential()
for layer in vgg16.layers:
    vgg16Model.add(layer)

# Add your custom Dense layer
vgg16Model.add(Dense(numberOfClass, activation="softmax"))

# Compile the model (specify optimizer, loss, metrics as needed)
vgg16Model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
hist_vgg16 = vgg16Model.fit_generator(train_data, 
                                      steps_per_epoch=1600 // batch_size, 
                                      epochs=10, 
                                      validation_data=test_data, 
                                      validation_steps=800 // batch_size)


# In[ ]:


hist_vgg16 = vgg16Model.fit(train_data, 
                            steps_per_epoch=1600 // batch_size, 
                            epochs=10, 
                            validation_data=test_data, 
                            validation_steps=800 // batch_size)


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Assuming you have defined numberOfClass and train_path and test_path earlier in your code
numberOfClass = 10  # Replace with the actual number of classes
train_path = "E://Users//kishore//Downloads//seg_train"
test_path = "E://Users//kishore//Downloads//seg_test"

# Using ImageDataGenerator for training data
train_data_gen = ImageDataGenerator(rescale=1./255)  # You can add more augmentations if needed
train_data = train_data_gen.flow_from_directory(train_path, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Using ImageDataGenerator for testing data
test_data_gen = ImageDataGenerator(rescale=1./255)
test_data = test_data_gen.flow_from_directory(test_path, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Build the VGG16 model
vgg16 = VGG16(include_top=False, input_shape=(224, 224, 3))  # Adjust input shape as needed

# Create a Sequential model and add the VGG16 layers
vgg16Model = Sequential()
for layer in vgg16.layers:
    vgg16Model.add(layer)

# Add your custom Dense layer
vgg16Model.add(Dense(numberOfClass, activation="softmax"))

# Compile the model (specify optimizer, loss, metrics as needed)
vgg16Model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
hist_vgg16 = vgg16Model.fit(train_data, 
                            steps_per_epoch=1600 // batch_size, 
                            epochs=10, 
                            validation_data=test_data, 
                            validation_steps=800 // batch_size)

# Plot the training accuracy and validation accuracy
plt.plot(hist_vgg16.history["accuracy"], label="accuracy")
plt.plot(hist_vgg16.history["val_accuracy"], label="validation accuracy")
plt.legend()
plt.show()


# In[ ]:


import json, codecs
with open("deneme.json","w") as f:
    json.dump(hist_vgg16.history, f)


# In[ ]:


import json

# Assuming you have executed the training code and hist_vgg16 is defined
# with the training history

# Save the training history to a JSON file
with open("deneme.json", "w") as f:
    json.dump(hist_vgg16.history, f)


# # Load Result 

# In[ ]:


with codecs.open("./deneme.json","r", encoding = "utf-8") as f:
    load_result = json.loads(f.read())


# In[ ]:


load_result


# In[ ]:


# Loss And Validation Loss
plt.plot(load_result["loss"], label = "training loss")
plt.plot(load_result["val_loss"], label = "validation loss")
plt.legend()
plt.show()


# In[ ]:


# Accuracy And Validation Accuracy
plt.plot(load_result["accuracy"], label = "accuracy")
plt.plot(load_result["val_accuracy"], label = "validation accuracy")
plt.legend()
plt.show()


# # VGG19 

# In[37]:


from keras.applications import VGG19

# Create an instance of the VGG19 model
vgg19 = VGG19()


# In[38]:


# Layers of vgg19
vgg19.summary()


# In[39]:


# Layers of vgg19 
vgg19_layer_list = vgg19.layers
for i in vgg19_layer_list:
    print(i)


# In[40]:


# add the layers of vgg16 in my created model.
vgg19Model = Sequential()
for i in range(len(vgg19_layer_list)-1):
    vgg19Model.add(vgg19_layer_list[i])


# In[41]:


# Finish version of my created model.
vgg19Model.summary()


# In[42]:


# Close the layers of vgg16
for layers in vgg19Model.layers:
    layers.trainable = False


# In[43]:


# Last layer
vgg19Model.add(Dense(numberOfClass, activation = "softmax"))


# In[44]:


# the final version of the model
vgg19Model.summary()


# In[45]:


# I create compile part.
vgg19Model.compile(loss = "categorical_crossentropy",
             optimizer = "rmsprop",
             metrics = ["accuracy"])


# # Visualize The Results Of Model

# In[9]:


# Loss And Validation Loss
plt.plot(hisy_vgg19.history["loss"], label = "training loss")
plt.plot(hisy_vgg19.history["val_loss"], label = "validation loss")
plt.legend()
plt.show()


# In[35]:


# Accuracy And Validation Accuracy
plt.plot(hisy_vgg19.history["accuracy"], label = "accuracy")
plt.plot(hisy_vgg19.history["val_accuracy"], label = "validation accuracy")
plt.legend()
plt.show()


# # Build a Multiclass image classification model with InceptionV3 and Mobilenet pretrained network.

# In[16]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as layers


# In[17]:


labels=pd.read_csv("C://Users//kishore//monkey_labels.txt")


# In[18]:


labels


# # Step 1: Pre-process and create train set

# In[19]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory("E://Users//kishore//Downloads//training",
                                                 target_size = (128,128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# # Step 2: pre-process and create test set

# In[20]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = train_datagen.flow_from_directory("E://Users//kishore//Downloads//validation",
                                                 target_size = (128,128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[21]:


import IPython.display as ipd

ipd.Image("E://Users//kishore//Downloads//training//training//n0//n0125.jpg")


# # Step 3: Import the pre- trained model

# In[22]:


from tensorflow.keras.applications.inception_v3 import InceptionV3
base_model = InceptionV3(input_shape = (128, 128, 3), include_top = False, weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False


# # Step 4: Add Flattening, hidden and output layers

# In[23]:


x=base_model.output
x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(10, activation='sigmoid')(x)

inception = tf.keras.models.Model(base_model.input, x)
inception.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# # inception.summary()

# In[24]:


inception.summary()


# # Step 6: Train and Test accuracy, loss plots

# In[28]:


pip install Inception_hist


# In[ ]:





# In[34]:


import matplotlib.pyplot as plt

# Assuming you have a dictionary containing the training history
# Replace this with your actual history
Inception_hist = {
    'accuracy': [0.5, 0.6, 0.7, 0.8],
    'val_accuracy': [0.4, 0.5, 0.6, 0.7],
}

# Summarize history for accuracy
plt.plot(Inception_hist['accuracy'], label='Train')
plt.plot(Inception_hist['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[31]:


import matplotlib.pyplot as plt

# Assuming you have trained your Inception model and saved the history in Inception_hist
# ...

# Create a sample history for demonstration purposes (replace this with your actual history)
Inception_hist = {
    'accuracy': [0.5, 0.6, 0.7, 0.8],
    'val_accuracy': [0.4, 0.5, 0.6, 0.7],
}

# Summarize history for accuracy
plt.plot(Inception_hist['accuracy'], label='Train')
plt.plot(Inception_hist['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:




