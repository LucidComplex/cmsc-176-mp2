#!/usr/bin/env python2
import numpy as np

from matplotlib import pyplot as plt
from logreg import (feature_normalize, load_data, sigmoid,
    gradient_descent_multi, predict)


def main():
    print 'Loading data...'
    X, y = load_data('ex2data2.txt')

    print 'First 10 examples from dataset:'
    print '\t\tX\t\t  y'
    for i in range(10):
        print '  {0}\t{1}'.format(X[i], y[i])

    print 'Plotting data...'
    plot_data(X, y)

    print 'Normalizing features...'
    X, mu, sigma = feature_normalize(X)

    print 'Adding polynomial features...'
    X = map_feature(X[:, 0], X[:, 1])

    initial_theta = np.zeros((X.shape[1], 1))

    lambda_ = 1

    print 'Computing cost...'
    cost, grad = cost_function_reg(initial_theta, X, y, lambda_)

    print 'Optimizing using gradient descent...'
    alpha = 0.05
    num_iters = 1000

    theta = np.zeros((X.shape[1], 1))
    theta, J_history = gradient_descent_multi_reg(
        X, y, theta, alpha, num_iters, lambda_)

    plt.plot(range(1, len(J_history) + 1), J_history, '-b')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()

    print 'Theta computed from gradient descent:'
    print theta

    chip_data = np.array([[-0.5, 0.7]])
    chip_data = (chip_data - mu) / sigma
    chip_data = np.insert(chip_data, 0, 1, axis=1)
    chip_data = map_feature(chip_data[:, 0], chip_data[:, 1])
    prob = sigmoid(chip_data.dot(theta))
    print 'For a microchip with test of -0.5 and 0.7, we predict an acceptance of {0}%'.format(prob[0, 0] * 100)

    print 'Computing accuracy:\n'
    p = predict(theta, X)
    accuracy = np.mean(y == p)
    print 'Train Accuracy: {0}%'.format(accuracy * 100)


def gradient_descent_multi_reg(X, y, theta, alpha, num_iters, lambda_):
    # regularize J_history
    m = len(y)
    theta[0] = 0
    cost_regularization_term = lambda_ / (2 * m) * sum(theta ** 2)
    grad_regularization_term = lambda_ / m * theta

    theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)
    J_history = J_history - cost_regularization_term
    theta = theta - grad_regularization_term

    return theta, J_history


def plot_data(X, y):
    pos = y == 1
    neg = y == 0
    plt.plot(X[:, (0, )][pos], X[:, (1, )][pos], 'b+', label='y = 1')
    plt.plot(X[:, (0, )][neg], X[:, (1, )][neg], 'yo', label='y = 0')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(loc='upper right')
    plt.show()
    plt.gcf().clear()


def map_feature(X_one, X_two):
    X_one = X_one.reshape(len(X_one), 1)
    X_two = X_two.reshape(len(X_two), 1)
    degree = 6
    out = np.ones((len(X_one), 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            new_feature = (X_one ** (i - j)) * (X_two ** j)
            out = np.append(out, new_feature,axis=1)
    return out


def cost_function_reg(theta, X, y, lambda_):
    h = sigmoid(X.dot(theta))
    log_h = np.log(h)
    one_minus_h = 1 - h
    log_one_minus_h = np.log(one_minus_h)
    vectorized_cost = -y * log_h - (1 - y) * log_one_minus_h
    m = len(y)
    J = np.sum(vectorized_cost) / m
    theta[0] = 0
    regularization = lambda_ / (2 * m) * sum(theta ** 2)
    J = J + regularization
    error = h - y
    grad = X.T.dot(error)
    grad = grad / m
    grad = grad + (lambda_ * theta) / m

    return J, grad


if __name__ == '__main__':
    main()
