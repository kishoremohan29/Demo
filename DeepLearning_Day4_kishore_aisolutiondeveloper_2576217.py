#!/usr/bin/env python
# coding: utf-8

# In[6]:


pip install torch


# In[9]:


pip install keras.wrappers


# In[10]:


pip install keras-tuner


# # Build a Neural network with Hyper-parameter fine tuning model

# In[11]:


from tensorflow import keras # importing keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() # loading the data using keras datasets api
x_train = x_train.astype('float32') / 255.0 # normalize the training images
x_test = x_test.astype('float32') / 255.0 # normalize the testing images


# In[12]:


model1 = keras.Sequential()
model1.add(keras.layers.Flatten(input_shape=(28, 28))) # flattening 28 x 28 
model1.add(keras.layers.Dense(units=512, activation='relu', name='dense_1')) # you have 512 neurons with relu activation
model1.add(keras.layers.Dropout(0.2)) # we added a dropout layer with the rate of 0.2
model1.add(keras.layers.Dense(10, activation='softmax')) # output layer, where we have total 10 classes


# In[13]:


model1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])
model1.fit(x_train, y_train, epochs=10, validation_split=0.2)


# In[ ]:


pip install model


# In[2]:


model1_eval = model.evaluate(img_test, label_test, return_dict=True)


# In[ ]:





# In[ ]:


import tensorflow as tf
import kerastuner as kt


# In[16]:


def model_builder(hp):
  '''
  Args:
    hp - Keras tuner object
  '''
  # Initialize the Sequential API and start stacking the layers
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))
  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(keras.layers.Dense(units=hp_units, activation='relu', name='dense_1'))
  # Add next layers
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(10, activation='softmax'))
  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
  return model


# In[4]:


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# Perform hypertuning
tuner.search(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[stop_early])
best_hp=tuner.get_best_hyperparameters()[0]


# In[6]:


# Build the model with the optimal hyperparameters
h_model = tuner.hypermodel.build(best_hps)
h_model.summary()
h_model.fit(x_train, x_test, epochs=10, validation_split=0.2)
Now, you can evaluate this model, 

h_eval_dict = h_model.evaluate(img_test,label_test, return_dict=True)


# In[1]:


# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters

# Load your dataset or use any available dataset
# For example, let's use the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model building function
def build_model(hp):
    model = keras.Sequential()
    
    # Flatten the input for the first layer
    model.add(layers.Flatten(input_shape=(28, 28)))
    
    # Tune the number of units in the first Dense layer
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    
    # Tune the number of hidden layers and their units
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32), activation='relu'))
    
    # Add the output layer
    model.add(layers.Dense(10, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Instantiate the tuner and perform the hyperparameter search
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='my_dir',  # Directory to store the results
    project_name='fashion_mnist_hyperband'  # Name of the project
)

# Display search space summary
tuner.search_space_summary()

# Perform the search
tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters and train it on the full dataset
model = tuner.hypermodel.build(best_hps)
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


# # Build an image classifier model with Pytorch

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def _init_(self):
        super(SimpleCNN, self)._init_()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc1(x)
        return x

# Download and prepare the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training the model
for epoch in range(5):  # Adjust the number of epochs as needed
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # Print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print("Training finished")

# Testing the model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on the test set: {100 * correct/total}%")


# In[2]:


pip install torchvision


# In[ ]:




