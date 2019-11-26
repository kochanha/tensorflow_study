# Lab 4 Multi-variable linear regression
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
tf.set_random_seed(777)  # for reproducibility

data = np.array([
          [73., 80., 75., 152.],
          [93., 88., 93., 185.],
          [89., 91., 90., 180.],
          [96., 98., 100., 196.],
          [73., 66., 70., 142.]
          ], dtype=np.float32)


# placeholders for a tensor that will be always fed.
X=data[:,:-1]
Y=data[:,[-1]]

W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

learning_rate = 0.000001

def predict(X):
    return tf.matmul(X,W)+b

n_epochs = 2000
for i in range(n_epochs+1):
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(predict(X)-Y)))

    W_grad, b_grad = tape.gradient(cost, [W,b])

    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i%100==0:
        print("{:5} | {:10.4f}".format(i, cost.numpy()))
