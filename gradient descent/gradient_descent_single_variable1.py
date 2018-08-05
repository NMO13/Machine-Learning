import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        #print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta


def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y

# gen 100 points with a bias of 25 and 10 variance as a bit of noise
x, y = genData(100, 25, 10)

df = pd.read_csv('./ex1data1.txt', names=['x','y'])
x = df['x']
y = df['y']
m = len(x)
n = 2
x1 = np.array(list(zip(np.ones(len(x)), x)))
#m, n = np.shape(x)
numIterations= 10000
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent(x1, y, theta, alpha, m, numIterations)
print(theta)

def f(l):
    return theta[0]+ l * theta[1]

plt.scatter(x, y,  color='black')
plt.plot(x, f(x), color='blue', linewidth=3)
plt.show()