import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data", one_hot=True)

def create_placeholder(n_H, n_W, n_C, n_y):
    '''
    n_H: the hight of image
    n_W: the width of image
    n_C: the image's channal numbers
    n_y: the class we need to classify
    '''
    X = tf.placeholder(tf.float32, shape = [None, n_H, n_W, n_C], name = "X")
    Y = tf.placeholder(tf.float32, shape = [None, n_y], name = "Y")

    return X, Y

def initialize_parameters():
    W1 = tf.get_variable(name = "W1", dtype = tf.float32, shape = [5, 5, 1, 6],
                        initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    '''
    W1_1 average pool ksize = (1, 2, 2, 1) strides = 2
    '''
    W2 = tf.get_variable(name = "W2", dtype = tf.float32, shape = [5, 5, 6, 16],
                        initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    '''
    W2_2 average pool ksize = (1, 2, 2, 1) strides = 2
    '''
    parameters = {
        "W1": W1,
        "W2": W2
    }

    return parameters

def forward_propagation(X, parameters):
    '''
    Implement the forward propagation for the model:
    (32, 32, 1) -> Conv2D(5*5, strides = 1) -> Avg Pool(2*2, strides = 2) -> Conv2D(5*5, strides = 1) -> Avg Pool(2*2, strides = 2)
    ->Flatten1(400) -> fullconnected(84) -> softmax(10)
    '''
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    Z1 = tf.nn.conv2d(input = X, filter = W1, strides = (1, 1, 1, 1), padding = "VALID")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.avg_pool(value = A1, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "VALID")

    Z2 = tf.nn.conv2d(input = P1, filter = W2, strides = (1, 1, 1, 1), padding = "VALID")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.avg_pool(value = A2, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "VALID")
    
    Z3 = tf.contrib.layers.flatten(P2)
    A3 = tf.nn.relu(Z3)

    Z4 = tf.contrib.layers.fully_connected(A3, 84)
    A4 = tf.nn.relu(Z4)

    Z5 = tf.contrib.layers.fully_connected(A4, 10, activation_fn = None)

    return Z5

def compute_cost(Z5, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z5, labels = Y))
    return cost

def model(mnist, learning_rate = 0.001, batch_size = 128, training_iters = 200000,print_cost = True):
    n_H = 28
    n_W = 28
    n_C = 1
    n_y = 10
    X, Y = create_placeholder(n_H, n_W, n_C, n_y)
    costs = []
    
    parameters = initialize_parameters()

    Z5 = forward_propagation(X, parameters)

    cost = compute_cost(Z5, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(Z5, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        step = 1

        while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = batch_x.reshape([128, 28, 28, 1])
            sess.run(optimizer, feed_dict = {X: batch_x, Y: batch_y})

            if step % 10 == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict = {X: batch_x, Y: batch_y})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
                costs.append(loss)
            step += 1

        sess.run(parameters)
        print("Optimization finished!")

        print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={
                                            X: mnist.test.images[:1024].reshape(1024, 28, 28, 1),
                                            Y: mnist.test.labels[:1024]
                                            }))
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (pre hundreds)')
        plt.title("Learning rate is " + str(learning_rate))
        plt.show()
    
    return parameters

parameters = model(mnist)
