import numpy as np


class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By defaul it's 1.0."""

    def __init__(self, inputs, bias=1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias)."""
        self.weights = (np.random.rand(inputs + 1) * 2) - 1
        self.bias = bias

    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        sum = np.dot(np.append(x, self.bias), self.weights)
        return self.sigmoid(sum)

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
                    print(" neurone :", j)
                    self.network[i].append(Perceptron(inputs=self.layers[i - 1], bias=self.bias))
            print( self.network[i])
        self.network = np.array([np.array(x) for x in self.network], dtype=object)
        self.values = np.array([np.array(x) for x in self.values], dtype=object)
        self.d = np.array([np.array(x) for x in self.d], dtype=object)


    def set_weights(self, w_init):
        """Set the weights.
           w_init is a list of lists with the weights for all but the input layer."""
        for i in range(len(w_init)):  # layers
            for j in range(len(w_init[i])):  # neurones
                self.network[i + 1][j].set_weights(w_init[i][j])

    def printWeights(self):
        print()
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                print("Layer", i, "Neuron", j, self.network[i][j].weights)
        print()

    def run(self, x):
        """Feed a sample x into the MultiLayer Perceptron."""
        x = np.array(x, dtype=object)
        self.values[0] = x
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i - 1])
        print("Output values for each neurones ")
        print(self.values)
        return self.values[-1]


mlp = MultiLayerPerceptron(layers=[2, 2, 1])
# '2 input, puis 2 neurones , sur le premier couche puis 1 sur la seconde couche de sortie

mlp.set_weights([[[-10, -10, 15], [15, 15, -10]], [[10, 10, -15]]])
print("---------------------weight by layer---------------------")
mlp.printWeights()
print("---------------------layer - 1----------------------")

print(mlp.layers[-1])
print("-------------------------------------------")
print("-------mlp-----00------------------")
print("pour  0 0 = {0:.10f}".format(mlp.run([0, 0])[0]))
print("-------mlp-----01------------------")
print("pour  0 1 = {0:.10f}".format(mlp.run([0,1]) [0]))
print("-------mlp-----10------------------")
print("pour  1 0 = {0:.10f}".format(mlp.run([1,0]) [0]))
print("-------mlp-----11------------------")
print("pour  1 1 = {0:.10f}".format(mlp.run([1,1]) [0]))