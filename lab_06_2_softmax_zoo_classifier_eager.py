# -*- coding: utf-8 -*-
"""lab-06-2-softmax_zoo_classifier-eager.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1D_BnhdXx0sYk3j180CPjZBlNAu0Tza0r

# Lab 06 Softmax Zoo_classifier-eager

* Softmax를 사용하여 Zoo 데이터를 활용하여 분류를 진행합니다.

### 기본 Library 선언 및 Tensorflow 버전 확인
"""

import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
tf.set_random_seed(777)  # for reproducibility
tfe = tf.contrib.eager

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.int32) #tf1.13.1에서는 np.int32, 이전에는 np.float32
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7  # 0 ~ 6

# Make Y data as onehot shape
Y_one_hot = tf.one_hot(list(y_data), nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

print(x_data.shape, Y_one_hot.shape)

#Weight and bias setting
W = tfe.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tfe.Variable(tf.random_normal([nb_classes]), name='bias')
variables = [W, b]

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
def logit_fn(X):
    return tf.matmul(X, W) + b

def hypothesis(X):
    return tf.nn.softmax(logit_fn(X))

def cost_fn(X, Y):
    logits = logit_fn(X)
    cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                           labels=Y)
    cost = tf.reduce_mean(cost_i)
    
    return cost

def grad_fn(X, Y):
    with tf.GradientTape() as tape:
        loss = cost_fn(X, Y)
        grads = tape.gradient(loss, variables)
        return grads
    
def prediction(X, Y):
    pred = tf.argmax(hypothesis(X), 1)
    correct_prediction = tf.equal(pred, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy

def fit(X, Y, epochs=1000, verbose=100):
    optimizer =  tf.train.GradientDescentOptimizer(learning_rate=0.1)

    for i in range(epochs):
        grads = grad_fn(X, Y)
        optimizer.apply_gradients(zip(grads, variables))
        if (i==0) | ((i+1)%verbose==0):
#             print('Loss at epoch %d: %f' %(i+1, cost_fn(X, Y).numpy()))
            acc = prediction(X, Y).numpy()
            loss = cost_fn(X, Y).numpy() 
            print('Steps: {} Loss: {}, Acc: {}'.format(i+1, loss, acc))

fit(x_data, Y_one_hot)