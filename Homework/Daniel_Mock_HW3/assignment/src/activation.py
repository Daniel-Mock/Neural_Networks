#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: activation.py

import numpy as np


def sigmoid(z):
    """The sigmoid function."""
    #e^x/e^x+1
    sig = np.exp(z)/(np.exp(z)+1)
    return sig

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    sig = sigmoid(z)
    sig_prime = sig*(1-sig)
    return sig_prime
