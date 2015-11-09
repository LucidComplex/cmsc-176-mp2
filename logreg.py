#!/usr/bin/env python2
import numpy as np

from math import e
from matplotlib import pyplot


def main():
    print 'Loading data...\n'
    X, y = load_data('ex2data1.txt')

    print 'First 10 examples from dataset:\n'
    print '\t\tX\t\t  y'
    for i in range(10):
        print '  {0}\t{1}'.format(X[i], y[i])

    print 'Normalizing features...\n'
    X, mu, sigma = feature_normalize(X)

    print 'Plotting data...\n'
    plot_data(X, y)

    print 'Computing cost and gradient...\n'
    m, n = X.shape
    X = np.insert(X, 0, 1, axis=1)
    initial_theta = np.zeros((n + 1, 1))
    cost, grad = cost_function(initial_theta, X, y)


def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = np.divide((X - mu), sigma)

    return X_norm, mu, sigma


def plot_data(X, y):
    pos = y == 1
    neg = y == 0
    pyplot.plot(X[:, (0, )][pos], X[:, (1, )][pos], 'b+', label='Admitted')
    pyplot.plot(X[:, (0, )][neg], X[:, (1, )][neg], 'yo', label='Not admitted')

    pyplot.xlabel('Exam 1 score')
    pyplot.ylabel('Exam 2 score')
    pyplot.legend(loc='upper right')
    pyplot.show()


def cost_function(theta, X, y):
    m = len(y)

    J = 0
    grad = np.zeros(theta.shape)
    h = X.dot(theta)


    return J, grad


def sigmoid(z):
    return 1 / (1 + e ** -z)


def load_data(filename):
    X = []
    y = []
    with open(filename, 'r') as fp:
        for line in fp:
            items = line.split(',')
            X.append(items[:-1])
            y.append([items[-1],])
        X = np.array(X, dtype=np.float)
        y = np.array(y, dtype=np.float)
    return X, y


if __name__ == '__main__':
    main()
