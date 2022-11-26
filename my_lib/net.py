import pickle
import numpy as np
from .utils import numerical_gradient 
from .layers import BatchNormalization, Dropout, Sigmoid, Relu, Affine, \
    SoftmaxWithLoss, Convolution, Pooling


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
        self.layers["Affine" + str(idx)] = Affine(self.params["W" + str(idx)], self.params["b" + str(idx)])
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
        y = np.argmax(y, axis=1)
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
            grads["W" + str(i)] = self.layers["Affine" + str(i)].dW + self.weight_decay_lambda * self.layers["Affine" + str(i)].W
            grads["b" + str(i)] = self.layers["Affine" + str(i)].db
        return grads

class MultiLayerNetExtend:
    def __init__(self, input_size, hidden_size_list, output_size, 
                activation="relu", weight_init_std="relu", weight_decay_lambda=0,
                use_dropout=False, dropout_ration=0.5, use_batchnorm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.params = {}

        self.__init_weight(weight_init_std)

        activation_layer = {"sigmoid": Sigmoid, "relu": Relu}
        self.layers = {}
        for i in range(1, self.hidden_layer_num + 1):
            self.layers["Affine" + str(i)] = Affine(self.params["W" + str(i)], self.params["b" + str(i)])

            if self.use_batchnorm:
                self.params["gamma" + str(i)] = np.ones(hidden_size_list[i - 1])
                self.params["beta" + str(i)] = np.zeros(hidden_size_list[i - 1])
                self.layers["BatchNorm" + str(i)] = BatchNormalization(self.params["gamma" + str(i)], self.params["beta" + str(i)])

            self.layers["Activation_function" + str(i)] = activation_layer[activation]()

            if self.use_dropout:
                self.layers["Dropout" + str(i)] = Dropout(dropout_ration)

        idx = self.hidden_layer_num + 1
        self.layers["Affine" + str(idx)] = Affine(self.params["W" + str(idx)], self.params["b" + str(idx)])
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

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        weight_decay = 0
        for i in range(1, self.hidden_layer_num + 2):
            W = self.params["W" + str(i)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t, train_flg=True)
        grads = {}
        for i in range(1, self.hidden_layer_num + 2):
            grads["W" + str(i)] = numerical_gradient(loss_W, self.params["W" + str(i)])
            grads["b" + str(i)] = numerical_gradient(loss_W, self.params["b" + str(i)])
            if self.use_batchnorm and i != self.hidden_layer_num + 1:
                grads["gamma" + str(i)] = numerical_gradient(loss_W, self.params["gamma" + str(i)])
                grads["beta" + str(i)] = numerical_gradient(loss_W, self.params["beta" + str(i)]) 
        return grads

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        for i in range(1, self.hidden_layer_num + 2):
            grads["W" + str(i)] = self.layers["Affine" + str(i)].dW + self.weight_decay_lambda * self.layers["Affine" + str(i)].W
            grads["b" + str(i)] = self.layers["Affine" + str(i)].db
            if self.use_batchnorm and i != self.hidden_layer_num + 1:
                grads["gamma" + str(i)] = self.layers["BatchNorm" + str(i)].dgamma
                grads["beta" + str(i)] = self.layers["BatchNorm" + str(i)].dbeta 
        return grads


class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + (2 * filter_pad)) \
                            / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))
        
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], 
                                                                 filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        
        self.layers = {}
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], 
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = 0.0        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        return acc / x.shape[0]
    
    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db
        
        return grads
    
    def save_params(self, path='params.pkl'):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(path, 'wb') as f:
            pickle.dump(params, f)
        print("Save network parameters")
            
    def load_params(self, path='params.pkl'):
        with open(path, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]
        print("Load network parameters")
