# Source : http://python3.codes/neural-network-python-part-1-sigmoid-function-gradient-descent-backpropagation/
import numpy as np

epochs = 20  # Number of iterations
inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 2, 1
L = .1  # learning rate

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])


def sigmoid(x): return 1 / (1 + np.exp(-x))  # activation function


def sigmoid_(x): return x * (1 - x)  # derivative of sigmoid


# weights on layer inputs
Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))  # 2 sur 2
Wz = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))  # 2 sur 1
print("wh", Wh)
print("wz", Wz)
for i in range(epochs):
    H = sigmoid(np.dot(X, Wh))  # hidden layer results

    Z = np.dot(H, Wz)  # output layer, no activation

    print("----------Z--------------")  # 4 weighs
    print(Z)
    print("-------------------------")  # 2 weighs
    E = Y - Z  # how much we missed (error)  bizarre pas de fonction d'activation
    print("E error", E)
    dZ = E * L  # delta Z multiplied by learning rate
    print("dZ", dZ)
    Wz += H.T.dot(dZ)  # update output layer weights
    print("--------------------------------------------")
    print("--------------------------------------------")
    print("H.T", H.T)
    print("dZ", dZ)
    print("res", H.T.dot(dZ))
    print("--------------------------------------------")
    print("--------------------------------------------")
    print("----Wz modified-----------")
    print("Wz", Wz)

    dH = dZ.dot(Wz.T) * sigmoid_(H)  # delta H
    Wh += X.T.dot(dH)  # update hidden layer weights
    print("-------dh------------")
    print(dH)
    print("--------Wh-----------")
    print(Wh)

print("Z", Z)  # what have we learnt?
print("---------weight output layer----------")
print("Wz", Wz)
print("----------weight hidden layer---------")
print("Wh", Wh)