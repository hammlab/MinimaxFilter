import numpy as np


def sigmoid(x):
        
    indp = np.where(x>=0)
    indn = np.where(x<0)
    tx = np.zeros(x.shape)
    tx[indp] = 1./(1.+np.exp(-x[indp]))
    tx[indn] = np.exp(x[indn])/(1.+np.exp(x[indn]))
    return tx


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def KL_divergence(x, y):
    return x * (np.log(x+1E-20)-np.log(y+1E-20)) + (1 - x) * (np.log(1 - x+1E-20) - np.log(1 - y+1E-20))


def initialize(hidden_size, visible_size):
    r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
    W1 = np.random.random((hidden_size, visible_size)) * 2 * r - r
    W2 = np.random.random((visible_size, hidden_size)) * 2 * r - r

    b1 = np.zeros(hidden_size, dtype=np.float64)
    b2 = np.zeros(visible_size, dtype=np.float64)

    theta = np.concatenate((W1.reshape(hidden_size * visible_size),
                            W2.reshape(hidden_size * visible_size),
                            b1.reshape(hidden_size),
                            b2.reshape(visible_size)))

    return theta


def sparse_autoencoder_cost(theta, visible_size, hidden_size,
                            lambda_, sparsity_param, beta, data):

    W1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    W2 = theta[hidden_size * visible_size:2 * hidden_size * visible_size].reshape(visible_size, hidden_size)
    b1 = theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
    b2 = theta[2 * hidden_size * visible_size + hidden_size:]

    m = data.shape[1]

    z2 = W1.dot(data) + np.tile(b1, (m, 1)).transpose()
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + np.tile(b2, (m, 1)).transpose()
    h = sigmoid(z3)

    cost = np.sum((h - data) ** 2) / (2 * m) + \
           (lambda_ / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))# + \

    sparsity_delta = 0

    delta3 = -(data - h) * sigmoid_prime(z3)
    delta2 = (W2.transpose().dot(delta3) + beta * sparsity_delta) * sigmoid_prime(z2)
    W1grad = delta2.dot(data.transpose()) / m + lambda_ * W1
    W2grad = delta3.dot(a2.transpose()) / m + lambda_ * W2
    b1grad = np.sum(delta2, axis=1) / m
    b2grad = np.sum(delta3, axis=1) / m

    grad = np.concatenate((W1grad.reshape(hidden_size * visible_size),
                           W2grad.reshape(hidden_size * visible_size),
                           b1grad.reshape(hidden_size),
                           b2grad.reshape(visible_size)))

    return cost, grad


def sparse_autoencoder(theta, hidden_size, visible_size, data):
    W1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    b1 = theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]

    m = data.shape[1]

    z2 = W1.dot(data) + np.tile(b1, (m, 1)).transpose()
    a2 = sigmoid(z2)

    return a2


def sparse_autoencoder_linear_cost(theta, visible_size, hidden_size,
                                   lambda_, sparsity_param, beta, data):

    W1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    W2 = theta[hidden_size * visible_size:2 * hidden_size * visible_size].reshape(visible_size, hidden_size)
    b1 = theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
    b2 = theta[2 * hidden_size * visible_size + hidden_size:]

    m = data.shape[1]

    z2 = W1.dot(data) + np.tile(b1, (m, 1)).transpose()
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + np.tile(b2, (m, 1)).transpose()
    h = z3

    cost = np.sum((h - data) ** 2) / (2 * m) + \
           (lambda_ / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2)) 


    sparsity_delta = 0.

    delta3 = -(data - h)
    delta2 = (W2.transpose().dot(delta3) + beta * sparsity_delta) * sigmoid_prime(z2)
    W1grad = delta2.dot(data.transpose()) / m + lambda_ * W1
    W2grad = delta3.dot(a2.transpose()) / m + lambda_ * W2
    b1grad = np.sum(delta2, axis=1) / m
    b2grad = np.sum(delta3, axis=1) / m

    grad = np.concatenate((W1grad.reshape(hidden_size * visible_size),
                           W2grad.reshape(hidden_size * visible_size),
                           b1grad.reshape(hidden_size),
                           b2grad.reshape(visible_size)))

    return cost, grad


