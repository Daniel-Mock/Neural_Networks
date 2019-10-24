#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(3, kernel_size=(3, 3), input_shape=(500,)))