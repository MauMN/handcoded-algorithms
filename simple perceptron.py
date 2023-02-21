import numpy as np
import matplotlib.pyplot as plt

w = np.random.rand(1,3) * 10
w_1 = np.round(w[0][0], 1)
w_2 = np.round(w[0][1], 1)
b = np.round(w[0][2], 1)

x = [ [-1,-1], [-1,1], [1,-1], [1,1]]
x_array = np.asarray(x)
x_1 = x[0][0]

yd = [-1, -1, -1, 1]
yd = np.asarray(yd)


def signo(h):
    if h >= 0:
        return 1
    else:
        return -1
    
plt.title('Input', fontsize=20)
plt.scatter(x_array[:,0], x_array[:,1])
plt.grid()
plt.show()
    
error = np.array([0,0,0,0])    
for i in range(len(x)):    
    h = np.dot(np.asarray([w_1, w_2]) , x[i])  + b
    y = signo(h)
    error[i] = (yd[i] - y)**2
    E = 0.5*np.sum(error)

learning_rate = 0.05


max_iter = 1000
j = 1
valores = [[w_1, w_2, b]]
while j < max_iter and E != 0:
    for i in range(len(x)):
        h = np.dot(np.asarray([w_1, w_2]) , x[i])  + b
        y = signo(h)
        e_signo = (yd[i] - y)
        error[i] = (yd[i] - y) ** 2
        w_1 = w_1 + learning_rate * e_signo * x[i][0]
        w_2 = w_2 + learning_rate * e_signo * x[i][1]
        b = b + learning_rate * e_signo

    valores = [[w_1, w_2, b]]
    E = 0.5 * np.sum(error)
    print('errors', E)
    j = j + 1