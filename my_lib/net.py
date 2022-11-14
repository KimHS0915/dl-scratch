import numpy as np
from my_lib.utils import numerical_gradient, Sigmoid, Relu, Affine, SoftmaxWithLoss

class MultiLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size, 
                activation="relu", weight_init_std="relu", weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        self.__init_weight(weight_init_std)

        activation_layer = {"sigmoid": Sigmoid, "relu": Relu}
        self.layers = {}
        for i in range(1, self.hidden_layer_num + 1):
            self.layers["Affine" + str(i)] = Affine(self.params["W" + str(i)], self.params["b" + str(i)])
            self.layers["Activation_function" + str(i)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers["Affine" + str(idx)] = Affine(self.params["W" + str(idx)], self.params["b", str(idx)])
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for i in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ("relu", "he"):
                scale = np.sqrt(2.0 / all_size_list[i - 1])
            elif str(weight_init_std).lower() in ("sigmoid", "xavier"):
                scale = np.sqrt(1.0 / all_size_list[i - 1])
            self.params["W" + str(i)] = scale * np.random.randn(all_size_list[i - 1], all_size_list[i])
            self.params["b" + str(i)] = np.zeros(all_size_list[i])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
            return x

    def loss(self, x, t):
        y = self.predict(x)
        weight_decay = 0
        for i in range(1, self.hidden_layer_num + 2):
            W = self.params["W" + str(i)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, aixs=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        for i in range(1, self.hidden_layer_num + 2):
            grads["W" + str(i)] = numerical_gradient(loss_W, self.params["W" + str(i)])
            grads["b" + str(i)] = numerical_gradient(loss_W, self.params["b" + str(i)])
        return grads

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        for i in range(1, self.hidden_layer_num + 2):
            grads["W" + str(i)] = self.layers["Affine" + str(i)].dw + self.weight_decay_lambda * self.layers["Affine" + str(i)].w
            grads["b" + str(i)] = self.layers["Affine" + str(i)].db
        return grads
