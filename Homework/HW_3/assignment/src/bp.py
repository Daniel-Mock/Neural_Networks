#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime

def backprop(x, y, biases, weightsT, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and transposed weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_wT): tuple containing the gradient for all the biases
                and weightsT. nabla_b and nabla_wT should be the same shape as
                input biases and weightsT
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_wT = [np.zeros(wT.shape) for wT in weightsT]

    ### Implement here
    # feedforward
    # Here you need to store all the activations of all the units
    # by feedforward pass
    ###

    #a = [np.matmul(weightsT[0],x) + biases[0]]
    #activations = [sigmoid(a[0])]
    a_k = []
    next_h = []
    next_h.append(x)

    for j in range(1,num_layers):
        a_k.append(np.dot(weightsT[j-1],next_h[j-1]) + biases[j-1])
        next_h.append(sigmoid(a_k[j-1]))
    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer

    #for k in range(1,num_layers - 1):
    #a.append(np.matmul(weightsT[k],a[k-1]) + biases[k])
    #activations.append(sigmoid(a[k]))


    delta = (cost).df_wrt_a(next_h[-1], y)




    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    ###

    #gradient w/ respect to the last layer
    #nabla_b[-1] = delta
    #nabla_wT[-1] = np.matmul(delta,np.transpose(activations[-2]))
    #G = np.matmul(weightsT[-1], delta)


    nabla_b[-1] = delta
    nabla_wT[-1] = np.dot(delta,np.transpose(next_h[-2]))
    for i in range(2,num_layers):
        delta = np.dot(np.transpose(weightsT[-i+1]),delta) * sigmoid_prime(a_k[-i])
        nabla_b[-i] = delta
        nabla_wT[-i] = np.dot(delta,np.transpose(next_h[-i-1]))



    return (nabla_b, nabla_wT)
