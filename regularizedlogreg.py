#!/usr/bin/env python2
import numpy as np

from matplotlib import pyplot as plt
from logreg import feature_normalize, load_data


def main():
    print 'Loading data...'
    X, y = load_data('ex2data2.txt')

    print 'First 10 examples from dataset:'
    print '\t\tX\t\t  y'
    for i in range(10):
        print '  {0}\t{1}'.format(X[i], y[i])

    print 'Normalizing features...'
    X, mu, sigma = feature_normalize(X)

    print 'Plotting data...'
    plot_data(X, y)

    print 'Adding polynomial features...'
    X = map_feature(X[:, 0], X[:, 1])


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
    end = 0
    for i in range(1, degree + 1):
        for j in range(i):
            out[end] = (X_one ** (i - j)) * (X_two ** j)
            end += 1

    print out
    return out


if __name__ == '__main__':
    main()
