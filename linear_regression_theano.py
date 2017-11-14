#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:12:26 2017

@author: amit

Linear Regression using Theano
"""

import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
rng = numpy.random

#Training Data
X_train = numpy.asarray([3,4,5,6.1,6.3,2.88,8.89,5.62,6.9,1.97,8.22,9.81,4.83,7.27,5.14,3.08])
Y_train = numpy.asarray([0.9,1.6,1.9,2.9,1.54,1.43,3.06,2.36,2.3,1.11,2.57,3.15,1.5,2.64,2.20,1.21])
X_test = numpy.linspace(0,10,10)

for steps in range (1, 1000):
    # m is the weight or theta
    m_value = rng.randn()
    
    # C is the intercept or constant
    c_value = rng.randn()
    
    m = theano.shared(m_value,name ='m')
    c = theano.shared(c_value,name ='c')
    
    x = T.vector('x')
    y = T.vector('y')
    
    num_samples = X_train.shape[0]
    
    
    prediction = T.dot(x,m)+c
    cost = T.sum(T.pow(prediction-y,2))/(2*num_samples)
    
    gradm = T.grad(cost,m)
    gradc = T.grad(cost,c)


    learning_rate = 0.001
    training_steps = steps
    
    # updates (iterable over pairs (shared_variable, new_expression) List, tuple or dict.)
    train = theano.function([x,y],cost,updates = [(m,m-learning_rate*gradm),(c,c-learning_rate*gradc)])
    test = theano.function([x],prediction)
    
    
    for i in range(training_steps):
        costM = train(X_train,Y_train)
        #print(costM)
        
    print("Training Steps: "+str(steps))
    print("Slope :")
    print(m.get_value())
    print("Intercept :")
    print(c.get_value())
    
    
   
    Y_test = test(X_test)
    plt.scatter(X_train, Y_train, color = 'green')
    plt.scatter(X_test, Y_test,color = 'red')
    plt.show()