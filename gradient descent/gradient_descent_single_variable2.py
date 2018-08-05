from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from functools import reduce
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import pandas as pd

theta_0 = 500
theta_1 = 600

#X, y = make_blobs(n_samples=100, centers=1, n_features=2, center_box=(20.0, 20.0))
#x = X[:,0]
#y = X[:,1]

df = pd.read_csv('./ex1data1.txt', names=['x','y'])
x = df['x']
y = df['y']

def h_theta(x_i):
    return theta_0 + x_i * theta_1


def linear_regression(X, y, m_current=0, b_current=0, epochs=10, learning_rate=0.0005):
    N = float(len(y))
    for i in range(epochs):
        y_current = (m_current * X) + b_current
        cost = sum([data ** 2 for data in (y - y_current)]) / N
        m_gradient = (2 / N) * sum(X * (y_current - y))
        b_gradient = (2 / N) * sum(y_current - y)
        m_current = m_current - (learning_rate * m_gradient)
        b_current = b_current - (learning_rate * b_gradient)
    return m_current, b_current, cost

def cost_function(x, y):
    def squaredSum(acc, ex):
        return (h_theta(ex[0]) - ex[1]) * (h_theta(ex[0]) - ex[1]) + acc
    m = len(x)
    data = zip(x, y)
    res = reduce(squaredSum, data, 0)
    return (res/(2*m))

def compute_cost_function(m, t0, t1, x, y):
    a = [(t0 + t1* x[i] - y[i])**2 for i in range(m)]
    return 1/2/m * sum(a)

# contour plot for 2 features
fig = plt.figure()
ax = fig.gca(projection='3d')

cost_values = []
r0 = np.arange(-10, 20, 0.5)
r1 = np.arange(-10, 20, 0.5)
r0, r1 = np.meshgrid(r0, r1)
for i in range(r0.shape[0] * r0.shape[1]):
    theta_0 = np.ravel(r0)[i]
    theta_1 = np.ravel(r1)[i]
    cost_values.append(compute_cost_function(len(x), theta_0, theta_1, x, y))

cost_values = np.array(cost_values).reshape(r0.shape)
ax.plot_surface(r0, r1, cost_values, cmap=plt.get_cmap('viridis'),
                       linewidth=0, antialiased=False)
theta_0 = 20
theta_1 = 20
ax.scatter(theta_0, theta_1, cost_function(x, y), c='black', marker='^', s=30)

# use SGD to minimize theta_0 and theta_1

def derivative_term(x, y, sum):
    m = len(x)
    data = zip(x, y)
    res = reduce(sum, data, 0)
    return (res/m)

def sum1(acc, ex):
    return (h_theta(ex[0]) - ex[1]) + acc

def sum2(acc, ex):
    return (ex[0] * (h_theta(ex[0]) - ex[1])) + acc

def gd():
    global theta_0
    global theta_1
    elements = []
    for i in range(0, 1500):
        alpha = 0.01
        #der_term_0 = derivative_term(x, y, sum1)
        #der_term_1 = derivative_term(x, y, sum2)
        #temp_0 = theta_0 - alpha * derivative_term(x, y, sum1)
        #temp_1 = theta_1 - alpha * derivative_term(x, y, sum2)

        #theta_0 = temp_0
        #theta_1 = temp_1
        #elements.append([temp_0, temp_1, cost_function(x, y)])

        grad0 = (2.0 / len(x)) * sum([(theta_0 + theta_1 * x[i] - y[i]) for i in range(len(x))])
        grad1 = (2.0 / len(x)) * sum([(theta_0 + theta_1 * x[i] - y[i]) * x[i] for i in range(len(x))])
        # update the theta_temp
        theta_0 = theta_0 - alpha * grad0
        theta_1 = theta_1 - alpha * grad1
        elements.append([theta_0, theta_1, compute_cost_function(len(x), theta_0, theta_1, x, y)])
    return np.array(elements)

elements = gd()
ax.scatter(elements[:,0], elements[:,1], elements[:,2], c='black', marker='^', s=30)
plt.show()


def f(l):
    return theta_0 + l * theta_1

plt.scatter(x, y,  color='black')
plt.plot(x, f(x), color='blue', linewidth=3)
plt.show()

# same algorithm as gd(), just a bit different implemented
theta_1, theta_0, cost = linear_regression(x, y, m_current=20, b_current=20, epochs=10000)
plt.scatter(x, y,  color='black')
plt.plot(x, f(x), color='blue', linewidth=3)
plt.show()










