#!/usr/bin/env python2
import numpy as np

from math import e
from matplotlib import pyplot as plt


def main():
    print 'Loading data...\n'
    X, y = load_data('ex2data1.txt')

    print 'First 10 examples from dataset:\n'
    print '\t\tX\t\t  y'
    for i in range(10):
        print '  {0}\t{1}'.format(X[i], y[i])

    print 'Plotting data...\n'
    plot_data(X, y)

    print 'Normalizing features...\n'
    X, mu, sigma = feature_normalize(X)

    print 'Computing cost and gradient...\n'
    m, n = X.shape
    X = np.insert(X, 0, 1, axis=1)
    initial_theta = np.zeros((n + 1, 1))
    cost, grad = cost_function(initial_theta, X, y)
    print 'Cost at initial theta (zeros):\n{0}'.format(cost)
    print 'Gradient at initial theta (zeros):\n{0}'.format(grad)

    print 'Optimizing using gradient descent...\n'
    alpha = 0.1
    num_iters = 1000
    theta = np.zeros((3, 1))
    theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)

    print 'Plotting convergence graph...\n'
    plt.plot(range(1, num_iters + 1), J_history)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()

    print 'Theta computed from gradient descent: \n'
    print theta

    print 'Prediction accuracy:\n'
    student_score = np.array([[45, 85]])
    student_score = student_score - mu
    student_score = student_score / sigma
    student_score = np.insert(student_score, 0, 1, axis=1)
    prob = sigmoid(student_score.dot(theta))
    print '  For a student with scores 45 and 85, we predict an admission probability of {0}%'.format(prob[0, 0] * 100)

    print 'Computing accuracy:\n'
    p = predict(theta, X)
    accuracy = np.mean(y == p)
    print 'Train Accuracy: {0}%'.format(accuracy * 100)


def predict(theta, X):
    h = sigmoid(X.dot(theta))
    predicted = h >= 0.5
    return predicted


def feature_normalize(X):
    mu = np.mean(X, axis=0).reshape((1, X.shape[1]))
    sigma = np.std(X, axis=0).reshape((1, X.shape[1]))
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


def plot_data(X, y):
    pos = y == 1
    neg = y == 0
    plt.plot(X[:, (0, )][pos], X[:, (1, )][pos], 'b+', label='Admitted')
    plt.plot(X[:, (0, )][neg], X[:, (1, )][neg], 'yo',
        label='Not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(loc='upper right')
    plt.show()
    plt.gcf().clear()


def cost_function(theta, X, y):
    h = sigmoid(X.dot(theta))

    log_h = np.log(h)
    one_minus_h = 1 - h
    log_one_minus_h = np.log(one_minus_h)
    vectorized_cost = -y * log_h - (1 - y) * log_one_minus_h
    m = len(y)
    J = np.sum(vectorized_cost) / m

    grad = np.zeros(theta.shape)
    error = h - y
    grad = np.sum(error * X, axis=0)
    grad = grad.reshape(theta.shape)
    grad = grad / m

    return J, grad


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        cost, grad = cost_function(theta, X, y)
        delta = alpha * grad
        theta = theta - delta
        J_history[i] = cost

    return theta, J_history


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
