import numpy as np
import matplotlib.pyplot as plt

x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
vd = np.array([[-1], [1], [1], [-1]])

# Number of inputs
n_x = 2
# Number of neurons in output layer
n_y = 1
# Number of neurons in hidden layer
n_h = 2
# Learning rate
lr = 0.01
# Define random seed
#np.random.seed(3)
# Define weight matrices for neural network
w1 = np.random.normal(loc=0, scale=1, size=[n_x, n_h])  # Weight matrix for hidden layer
w2 = np.random.normal(loc=0, scale=1, size=[n_h, n_y])  # Weight matrix for output layer
b1 = np.random.normal(loc=0, scale=1, size=(1, n_h))
b2 = np.random.normal(loc=0, scale=1, size=(1, n_y))

def tan(h):
    h = np.tanh(h)
    return h

# Forward propagation
h1 = np.dot(x, w1) + b1
v1 = tan(h1)  # g(h)
h2 = np.dot(v1, w2) + b2
v2 = tan(h2)

# Backward propagation
error = vd - v2
dh2 = (1 - tan(h2) ** 2) * error  # delta capa salida  g'(h) = 1 - tan(v2)**2
dw2 = v1.T.dot(dh2) * lr  # deltaW capa salida
dh1 = (1 - tan(h1) ** 2) * dh2.dot(w2.T) #delta capa intermedia
dw1 = x.T.dot(dh1) * lr   #deltaW capa intermedia

# Updating W & b
w1 = w1 + dw1
w2 = w2 + dw2

b1 = b1 + np.sum(dh1, axis=0, keepdims=True) * lr
b2 = b2 + np.sum(dh2, axis=0, keepdims=True) * lr

E = 0.5 * np.sum(error)**2

plotX = []
plotY= []

max_iter = 1000
j = 1
while j < max_iter and E != 0:
    # Forward propagation
    h1 = np.dot(x, w1) + b1
    v1 = tan(h1)  # g(h)
    h2 = np.dot(v1, w2) + b2
    v2 = tan(h2)
    
    # Backward propagation
    error = vd - v2
    dh2 = (1 - tan(h2) ** 2) * error  # delta capa salida  g'(h) = 1 - tan(v2)**2
    dw2 = v1.T.dot(dh2) * lr  # deltaW capa salida
    dh1 = (1 - tan(h1) ** 2) * dh2.dot(w2.T)
    dw1 = x.T.dot(dh1) * lr
    
    # Updating W & b
    w1 = w1 + dw1
    w2 = w2 + dw2
    
    b1 = b1 + np.sum(dh1, axis=0, keepdims=True) * lr
    b2 = b2 + np.sum(dh2, axis=0, keepdims=True) * lr
    
    error2 = error**2
    E = 0.5 * np.sum(error2)
    
    
    plotX.append(j)
    plotY.append(E)
    print(f'Errors {E}')
    j = j + 1
    
  
plt.plot(plotX, plotY)
plt.title('Convergencia de error con backpropagation')
plt.xlabel("Iteracion")
plt.ylabel("Error")
plt.show()