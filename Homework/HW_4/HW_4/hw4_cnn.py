
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.activations import relu, softmax, tanh, sigmoid
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
import os
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


recorded_performance={}

#GET DATA FROM .GZ FILES
def get_input_data():

    with open(os.path.join(os.getcwd(), "train-images-idx3-ubyte.gz"), "rb") as f:
        training_data = extract_images(f)

    with open(os.path.join(os.getcwd(), "train-labels-idx1-ubyte.gz"), "rb") as f:
        training_labels = extract_labels(f)

    with open(os.path.join(os.getcwd(), "t10k-images-idx3-ubyte.gz"), "rb") as f:
        testing_data = extract_images(f)

    with open(os.path.join(os.getcwd(), "t10k-labels-idx1-ubyte.gz"), "rb") as f:
        testing_labels = extract_labels(f)

    return (training_data, training_labels), (testing_data, testing_labels)


#BUILD THE CNN MODEL
def cnn_model_build(input_shape, ACTICATION_FUNCTION_STR, LEARNING_RATE, DROP_PROBABILITY, NEURONS, num_classes):
    model = Sequential()

    model.add(Conv2D(
        filters=32,
        kernel_size=[3, 3],
        input_shape=input_shape,
        padding="same",
        activation=ACTICATION_FUNCTION_STR
    ))

    model.add(BatchNormalization())

    model.add(MaxPool2D(
        pool_size=[2, 2],
        strides=2
    ))


    model.add(Flatten())

    model.add(Dense(
        units=NEURONS,
        activation=ACTICATION_FUNCTION_STR
    ))

    model.add(Dropout(DROP_PROBABILITY))

    model.add(Dense(
        units=num_classes,
        activation=softmax
    ))

    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(lr=LEARNING_RATE),
        metrics=["accuracy"]
    )

    model.summary()
    return model

#TRAIN THE CNN MODEL
def cnn_model_train(model, training_data, training_labels, BATCH_SIZE, EPOCHS, validation_data, validation_labels, save_callback, tb_callback):
    history = model.fit(
        x=training_data,
        y=training_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(validation_data, validation_labels),
        shuffle=True,
        callbacks=[save_callback, tb_callback],
        verbose=0
    )

    training_accuracy = history.history["acc"]
    training_loss = history.history["loss"]
    validation_accuracy = history.history["val_acc"]
    validation_loss = history.history["val_loss"]
    return training_accuracy, training_loss, validation_accuracy, validation_loss

#TEST THE CNN MODEL
def cnn_model_test(model, testing_data, testing_labels):
    testing_loss, testing_accuracy = model.evaluate(
        x=testing_data,
        y=testing_labels,
        verbose=0
    )

    predictions = model.predict_proba(
        x=testing_data,
        batch_size=None,
        verbose=0
    )
    return testing_accuracy, testing_loss, predictions


def main(batch_size, drop_probability, activation_function):

    (training_data, training_labels), (testing_data, testing_labels) = get_input_data()

    validation_data = training_data[50000:]
    validation_labels = training_labels[50000:]
    training_data = training_data[:50000]
    training_labels = training_labels[:50000]

    # HYPERPARAMETERS
    EPOCHS = 5
    BATCH_SIZE = batch_size
    LEARNING_RATE = 0.001
    NEURONS = 256
    DROP_PROBABILITY = drop_probability
    ACTICATION_FUNCTION_STR = activation_function

    # input image dimensions
    img_rows, img_cols = 28, 28
    num_channels = 1
    input_shape = (img_rows, img_cols, num_channels)

    # output dimensions
    num_classes = 10


    folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"), str(ACTICATION_FUNCTION_STR))
    history_file = folder + "/cnn_" + str(ACTICATION_FUNCTION_STR) + ".h5"
    save_callback = ModelCheckpoint(filepath=history_file, verbose=1)
    tb_callback = TensorBoard(log_dir=folder)

    # Build, train, and test model
    model = cnn_model_build(input_shape, ACTICATION_FUNCTION_STR, LEARNING_RATE, DROP_PROBABILITY, NEURONS, num_classes)
    training_accuracy, training_loss, validation_accuracy, validation_loss = cnn_model_train(model, training_data, training_labels, BATCH_SIZE,
                                                                         EPOCHS, validation_data, validation_labels,
                                                                         save_callback, tb_callback)
    testing_accuracy, testing_loss, predictions = cnn_model_test(model, testing_data, testing_labels)

    # save test set results to csv
    predictions = np.round(predictions)
    predictions = predictions.astype(int)
    df = pd.DataFrame(predictions)
    df.to_csv("mnist.csv", header=None, index=None)



    # Final Loss and accuracy
    print("\n")
    print("Validation accuracy:")
    print(validation_accuracy)
    print("Final Loss: {:.4f}".format(testing_loss))
    print("Final Accuracy: {:.4f}".format(testing_accuracy))


if __name__ == '__main__':
    main(batch_size=64, drop_probability=.6, activation_function="relu")
