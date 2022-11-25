import numpy as np
from .optimizer import SGD, Momentum, AdaGrad, Adam


class Trainer:
    def __init__(self, net, x_train, y_train, x_test, y_test, 
                 epochs=20, batch_size=100, optimizer="SGD", optimizer_param={"lr": 0.01}, 
                 eval_sample_num_per_epoch=None, verbose=0):
        self.net = net
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_sample_num_per_epoch = eval_sample_num_per_epoch
        self.verbose = verbose

        optimizer_class_dict = {"sgd": SGD, "momentum": Momentum, 
                                "adagrad": AdaGrad, "adam": Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / self.batch_size, 1)
        self.max_iter = int(self.epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        grads = self.net.gradient(x_batch, y_batch)
        self.optimizer.update(self.net.params, grads)

        loss = self.net.loss(x_batch, y_batch)
        self.train_loss_list.append(loss)
        if self.verbose >= 2:
            print(f"train loss : {loss:.5f}")
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, y_train_sample = self.x_train, self. y_train
            x_test_sample, y_test_sample = self.x_test, self.y_test
            if not self.eval_sample_num_per_epoch is None:
                t = self.eval_sample_num_per_epoch
                x_train_sample, y_train_sample = self.x_train[:t], self.y_train[:t]
                x_test_sample, y_test_sample = self.x_test[:t], self.y_test[:t]

            train_acc = self.net.accuracy(x_train_sample, y_train_sample)
            test_acc = self.net.accuracy(x_test_sample, y_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose >= 1:
                print(f"Epoch : {self.current_epoch}, train acc : {train_acc:.5f}, test acc : {test_acc:.5f}")
        self.current_iter += 1

    def train(self):
        for _ in range(self.max_iter):
            self.train_step()
        test_acc = self.net.accuracy(self.x_test, self.y_test)
        if self.verbose >= 1:
            print(f"Final Test Accuracy\n\ttest acc : {test_acc:.5f}")
