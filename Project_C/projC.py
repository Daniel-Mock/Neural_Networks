# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 21:49:19 2019

@author: Jeff
"""

def import_dataset():
    import cv2
    import pandas
    import numpy as np
    
    training_folder = 'training\\'
    data_set_csv = pandas.read_csv('train_files.csv')
    
    filenames = data_set_csv['file_name'].tolist()
    labels = data_set_csv['annotation'].tolist()
    
    data_set = np.zeros((len(filenames), 480, 640), dtype=np.uint8)
    for x in range(0, len(filenames)):
        img = cv2.imread(training_folder+filenames[x], 0)
        data_set[x] = img
    
    return data_set, np.array(labels)

def build_nn_fit(training_data_set, validation_data_set):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.losses import sparse_categorical_crossentropy
    from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
    # Build NN model
    model = Sequential()
    model.add(Conv2D(3, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten()) # Flatten into 1D
#    model.add(Dense(5,activation='relu'))
    model.add(Dense(5,activation=tf.nn.softmax))
    model.summary()
    
    model.compile(optimizer='adam',
                  loss=sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    
    history = model.fit(x=training_data_set[0],y=training_data_set[1], batch_size=128, validation_data=validation_data_set)
#    history2 = model.fit(x=validation_data_set[0],y=validation_data_set[1], batch_size=10, validation_data=training_data_set)
    return history, model


train_images, train_labels = import_dataset()
train_images = train_images.reshape(train_images.shape[0], 480, 640, 1)
input_shape = (480, 640, 1)
train_images = train_images.astype('float32')
train_images /= 255

######################################
k = 0
k_step = int(len(train_images) * 0.6)
total_samples = len(train_images)
c_train_set = train_images[k:k+k_step,::]
c_train_labels = train_labels[k:k+k_step]

c_val_set = train_images[k+k_step:total_samples,::]
c_val_labels = train_labels[k+k_step:total_samples]

#####################################
print('Running Model Training')
hist, model = build_nn_fit([c_train_set, c_train_labels], (c_val_set,c_val_labels))


#import cv2
#cv2.imshow('1',train_images[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()