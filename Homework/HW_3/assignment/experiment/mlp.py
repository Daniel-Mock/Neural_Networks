#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

#DATA_PATH = '../../../data/'
DATA_PATH = 'C:/Users/User/.virtualenvs/User-jBUq-HwN/src/ECE_542/HW_3/ece542-csc591-Fall19/homework/HW_03/data'
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')


    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 20, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)

def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    #pdb.set_trace()
    # construct the network
    model = network2.Network([784, 20, 10])
    # train the network using SGD
    (evaluation_cost, evaluation_accuracy, \
     training_cost, training_accuracy) = model.SGD(
        training_data=train_data,
        epochs=100,
        mini_batch_size=128,
        eta=1e-3,
        lmbda = 0.0,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)


    accum = 0
    for i in range(len(test_data[0])):
        pred = np.argmax(model.feedforward(test_data[0][i]))
        if pred == test_data[1][i]:
            accum += 1
    test_accuracy = accum / len(test_data[0])
    print("Test accuracy: " + str(test_accuracy))

    plt.figure()
    plt.plot(training_cost)
    plt.title("Training Loss")
    plt.savefig("training_loss")
    plt.show()
    plt.clf()

    plt.figure()
    plt.plot(training_accuracy)
    plt.title("training Accuracy")
    plt.savefig("Training_accuracy")
    plt.show()
    plt.clf()

    plt.figure()
    plt.plot(evaluation_accuracy)
    plt.title("Evaluation Accuracy")
    plt.savefig("Evaluation_accuracy")
    plt.show()
    plt.clf()

    plt.figure()
    plt.plot(evaluation_cost)
    plt.title("evaluation Loss")
    plt.savefig("evaluation_loss")
    plt.show()
    plt.clf()


if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()
