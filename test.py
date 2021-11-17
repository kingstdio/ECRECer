import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras

print(tf.__version__)

for module in mpl, np, sklearn, tf, keras:
    print(module.__name__, module.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

def show_single_image(img_arr):
    plt.imshow(img_arr, cmap = "binary")
    plt.show()
    
#show_single_image(x_train[0])

def show_imgs(n_rows, n_cols, x_data, y_data, class_names):
    assert len(x_data) == len(y_data)
    assert n_rows * n_cols < len(x_data)
    
    plt.figure(figsize = (n_cols * 1.4, n_rows * 1.6))
    
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index+1)
            plt.imshow(x_data[index], cmap="binary", interpolation = 'nearest')
            plt.axis('off')
            plt.title(class_names[y_data[index]])
            
    plt.show()
    
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'shirt', 'Sneaker',
             'Bag', 'Ankle boot']

#show_imgs(3,5,x_train,y_train,class_names)

# tf.keras.models.Sequential()

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28, 28]))
model.add(keras.layers.Dense(300, activation = 'relu'))  #  relu: y = max(0, x)
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10,activation = 'softmax')) # softmax: y = [e^x1/sum e^x2/sum ...]

# reason for sparse: y->one_hot
model.compile(loss = "sparse_categorical_crossentropy",
             optimizer = "sgd",
             metrics = ["accuracy"])

model.layers

model.summary()

history = model.fit(x_train, y_train, epochs=10,
                   validation_data = (x_valid, y_valid))

