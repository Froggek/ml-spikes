import numpy as np


class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By default, it's 1.0."""

    def __init__(self, inputs, bias=1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias)."""
        self.weights = (np.random.rand(inputs + 1) * 2) - 1
        self.bias = bias

    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        local_sum = np.dot(np.append(x, self.bias), self.weights)
        return self.sigmoid(local_sum)

    def set_weights(self, w_init):
        """Set the weights. w_init is a python list with the weights."""
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        """Evaluate the sigmoid function for the floating point input x."""
        return 1 / (1 + np.exp(-x))


class MultiLayerPerceptron:
    """A multilayer perceptron class that uses the Perceptron class above.
       Attributes:
          layers:  A python list with the number of elements per layer.
          bias:    The bias term. The same bias is used for all neurons.
          eta:     The learning rate."""

    def __init__(self, layers, bias=1.0, eta=0.5):
        """Return a new MLP object with the specified parameters."""
        self.layers = np.array(layers, dtype=object)
        self.bias = bias
        self.eta = eta
        self.network = []  # The list of lists of neurons
        self.values = []  # The list of lists of output values
        self.d = []  # The list of lists of error terms (lowercase deltas)

        for i in range(len(self.layers)):  # for each layer
            print(" couche :", i)
            self.values.append([])
            self.d.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            self.d[i] = [0.0 for j in range(self.layers[i])]
            if i > 0:  # network[0] is the input layer, so it has no neurons
                for j in range(self.layers[i]):
                    print("Neuron: ", j)
                    self.network[i].append(Perceptron(inputs=self.layers[i - 1], bias=self.bias))

        self.network = np.array([np.array(x) for x in self.network], dtype=object)  # network of neurones
        self.values = np.array([np.array(x) for x in self.values], dtype=object)  # list de list de sorties
        self.d = np.array([np.array(x) for x in self.d], dtype=object)

    def set_weights(self, w_init):
        """Set the weights.
           w_init is a list of lists with the weights for all but the input layer."""
        for i in range(len(w_init)):  # layers
            for j in range(len(w_init[i])):  # neurons
                self.network[i + 1][j].set_weights(w_init[i][j])

    def printWeights(self):
        print()
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                print("Layer", i + 1, "Neuron", j, self.network[i][j].weights)
        print()

    def run(self, x):
        """Feed a sample x into the MultiLayer Perceptron."""
        x = np.array(x, dtype=object)
        self.values[0] = x
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i - 1])
        # print(" ouput values for each neurones ")
        # print(self.values)
        return self.values[-1]

    def bp(self, x, y):
        """Run a single (x,y) pair with the backpropagation algorithm."""

        x = np.array(x, dtype=object)
        y = np.array(y, dtype=object)

        # Backpropagation Step by Step:

        # STEP 1: Feed a sample to the network
        outputs = self.run(x)
        # print("-----------------------output---------------")
        # print( outputs)
        # print("-----------------------/output---------------")
        # STEP 2: Calculate the MSE
        error = (y - outputs)
        # print("-----------------------error---------------")
        # print(error)
        # print("-----------------------/error---------------")
        MSE = sum(error ** 2) / self.layers[-1]   # nb de neurone sur la couche de sortie
        # print("-----------------------MSE---------------")
        # print(MSE)
        # print("-----------------------/MSE---------------")
        # STEP 3: Calculate the output error terms
        self.d[-1] = outputs * (1 - outputs) * (error)
        # print("-----------------------error term---------------")
        # print(self.d[-1])
        # print("-----------------------/error term---------------")
        # STEP 4: Calculate the error term of each unit on each layer
        for i in reversed(range(1, len(self.network) - 1)):  # For each layer #i
            for h in range(len(self.network[i])):  # For each neuron #h in the layer #i
                fwd_error = 0.0
                for k in range(self.layers[i + 1]):
                    fwd_error += self.network[i + 1][k].weights[h] * self.d[i + 1][k]
                # print("-----------------------error term---------------")
                self.d[i][h] = self.values[i][h] * (1 - self.values[i][h]) * fwd_error  # delta = sigma(1 - sigma) * err
        print("----------d array------------------")
        print(self.d)
        print("----------/d array------------------")

        # STEPS 5 & 6: Calculate the deltas and update the weights
        for i in range(1, len(self.network)):  # layers
            for j in range(self.layers[i]):   # neuron #j in layer #i
                for k in range(self.layers[i - 1] + 1):  # input (weighs #k) neuron #j, layer #i
                    if k == self.layers[i - 1]:  # bias
                        delta = self.eta * self.d[i][j] * self.bias  # eta = learning rate ("LR")
                    else:
                        delta = self.eta * self.d[i][j] * self.values[i - 1][k]
                    self.network[i][j].weights[k] += delta
        return MSE


mlp = MultiLayerPerceptron(layers=[2, 2, 1])
# 2 input , puis 2 neurones , sur le premier couche puis 1 sur la seconde couche de sortie

print(" training neural network for XOR gate ....cost function value.....................")
listmse=[]
nb_epochs = 10000
for i in range(nb_epochs):  # epoch
    MSE = 0.0  # cost function
    MSE += mlp.bp([0, 0], [0])
    MSE += mlp.bp([0, 1], [1])
    MSE += mlp.bp([1, 0], [1])
    MSE += mlp.bp([1, 1], [0])
    MSE /= 4
#    if i % 100 == 0:
#        print('Cost funtion ',MSE)
#        listmse.append(MSE)

xpoints = np.arange(30)
print("--------------xpoints--------------------")
print(xpoints)
ypoints = np.array(listmse )
print("--------------ypoints--------------------")
print(ypoints)

# import matplotlib.pyplot as plt
# plt.plot(xpoints, ypoints)
# plt.plot(xpoints, ypoints,'o:r')
# plt.show()

"""The LOSS function (or error) is for a single training example,
 while the COST function is over the entire training set (or mini-batch for mini-batch gradient descent).
  Generally cost and loss functions are synonymous but cost function can contain regularization terms 
  in addition to loss function"""

print("-----------application--------------------")
print(" ------- weight -----")
print(mlp.printWeights())
print("-------mlp-----00------------------")
print("pour  0 0 = {0:.10f}".format(mlp.run([0,0]) [0]))
print("-------mlp-----01------------------")
print("pour  0 1 = {0:.10f}".format(mlp.run([0,1]) [0]))
print("-------mlp-----10------------------")
print("pour  1 0 = {0:.10f}".format(mlp.run([1,0]) [0]))
print("-------mlp-----11------------------")
print("pour  1 1 = {0:.10f}".format(mlp.run([1,1]) [0]))