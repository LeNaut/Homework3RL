import numpy as np
import math
import matplotlib.pyplot as plt

class SharedFunctions:
    # here are frequently used functions

    x = np.empty([1])
    n = 1
    opt_param = []
    data = []

    def load_data(self, string):
        """
        loads data from training_data.txt
        :return: matrix of training data as in file
        """
        data = []
        i = 0
        with open(string, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                for i in range(0, len(line), 6):
                    strpart = line[i:i+6]
                    data.append(float(strpart))
        data = np.ndarray(shape=(2, int(len(data)/2)), buffer=np.array(data))
        return data

    def get_data(self):
        """
        :return: dataset
        """
        return self.data

    def get_model_fitting(self, y):
        """
        gets best fitting learn value, iterative
        :return: new learned value vector
        """
        p_inv_part = self.get_feature_matrix().T @ self.get_feature_matrix()
        self.opt_param = np.linalg.inv(p_inv_part) @ self.get_feature_matrix().T @ y
        return self.opt_param

    def get_feature_matrix(self):
        """
        :return: complete feature matrix
        """
        feat_mat = np.zeros((self.x.shape[0], self.n))
        for i in range(self.n):
            for x in self.x:
                feat_mat[np.where(self.x == x), i] = float(self.get_feature(x, i))
        return feat_mat

    def get_y_vector(self):
        """
        :return: y vector
        """
        y = []
        for x in self.x:
            y.append(self.get_func(x))
        return y

    def get_feature(self, x, i):
        """
        returns one element of the feature matrix
        :param x: point on x axis
        :param i: n selected
        :return: result of element
        """
        return math.sin(2 ** i * x)

    def get_func(self, x):
        """
        calculates learned result, which is also the y value of the solution
        :param x: x point
        :return: y value
        """
        func = 0
        for i in range(self.n):
            func += self.get_feature(x, i) * self.opt_param[i]
        return func

    def run(self):
        return

    def get_x(self):
        """
        :return: used x values as vector
        """
        return self.x

    def get_y_for_x_vector(self, x):
        y = []
        for p in x:
            y.append(self.get_func(p))
        return np.array(y)

    def get_mean(self, x):
        """
        :param x:
        :return: mean of data
        """
        sum = 0.0
        for i in range(self.n):
            sum = self.get_feature(x, i)
        return sum / self.n

    def get_rmse(self, x_points):
        """
        :return: RMSE of trained function
        """
        diff = 0
        for x in x_points:
            diff += (self.get_mean(x) - self.get_func(x))**2
        return math.sqrt(1 / (self.n * x_points.size) * diff)

    def get_rmse_validation_data(self):
        """
        :return: rsme of validation data
        """
        val_data = self.load_data('data/validation_data.txt')[1]
        val_axis = self.load_data('data/validation_data.txt')[0]

        diff = 0
        for x in val_axis:
            diff += (val_data[np.where(val_axis == x)] - self.get_func(x)) ** 2
        return math.sqrt(1 / len(val_data) * diff)

    def get_rmse_own_data(self):
        """
        :return: rsme of own data
        """
        diff = 0
        for x in self.x:
            diff += (self.data[np.where(self.x == x)] - self.get_func(x)) ** 2
        return math.sqrt(1 / len(self.data) * diff)


class SinFeatures(SharedFunctions):
    #not used in this task

    x = np.empty([1])
    n = 1 #number of features
    opt_param = [] # parameter vector to optimize

    def __init__(self, n):
        """
        :param n: count of features
        """
        super().__init__()
        self.x = np.arange(0, 6.02, 0.02)
        self.n = n
        self.opt_param = np.ones(n)
        return

    def get_learn_param(self):
        """
        :return: learned values as vector
        """
        return self.opt_param

    def get_mean(self, x):
        """
        :param x:
        :return: mean of data
        """
        sum = 0.0
        for i in range(self.n):
            sum = self.get_feature(x, i)
        return sum / self.n

    def run(self):
        """
        learns value
        :return: best learned y vector
        """
        super().get_model_fitting(super().get_y_vector())
        return super().get_y_vector()

class ImportedTraining(SharedFunctions):
    # used in task c - e

    data = [] #y pos of data
    x = [] #x pos of data
    n = 1
    opt_param = []

    def __init__(self, n):
        """
        loads data
        :param n: features
        """
        self.data = super().load_data('data/training_data.txt')[1]
        self.x = super().load_data('data/training_data.txt')[0]
        self.n = n
        self.opt_param = np.ones(n)
        return

    def run(self):
        """
        :return: optimized y vector
        """
        super().get_model_fitting(self.data)
        return super().get_y_vector()

class LooValidation(ImportedTraining):
    #used in task f

    data = np.zeros(1) #y pos data, which will be manipulated
    x = []# x pos data, which will be manipulated
    n = 1
    opt_param = []
    saved_data = np.zeros(1)#y pos data, (will not be manipulated)
    saved_x = []#x pos data, (will not be manipulated)
    valdata = 0#y pos val. data
    valx = 0#x pos val. data

    def __init__(self, n):
        super().__init__(n)
        self.saved_data = self.data
        self.saved_x = self.x
        self.valdata = self.load_data('data/validation_data.txt')[1]
        self.valx = self.load_data('data/validation_data.txt')[1]

    def manipulate_data(self, index):
        """
        removes one index out of data
        :param index: index which will get removed
        :return: manipulated data
        """
        if index == 0:
            self.data = self.saved_data[1:]
            self.x = self.saved_x[1:]
        elif index == self.saved_data.size-1:
            self.data = self.saved_data[:self.saved_data.size-1]
            self.x = self.saved_x[:self.saved_data.size-1]
        else:
            self.data = np.concatenate((np.array(self.saved_data[:index]), np.array(self.saved_data[index+1:])))
            self.x = np.concatenate((self.saved_x[:index], self.saved_x[index + 1:]))
        return self.data

    def get_rmse_llo_index(self, index):
        """
        :param index: index of the data point which was removed before
        :return: deviation from a data point (Y)
        """
        return np.linalg.norm(self.saved_data[index] - super().get_func(self.saved_x[index]))

    def get_rmse_llo_index_val(self, index):
        """
        :param index: index of the data point which was removed before
        :return: deviation from a data point (Y), validationset used
        """
        return np.linalg.norm(self.valdata[index] - super().get_func(self.valx[index]))

    def llo(self):
        """
        computes llo by training with one missing point and then validates this point. it does this for every point
        :return: llo mean and variance
        """
        llo_points = []
        training_var = []
        for i in range(self.saved_data.size):
            self.manipulate_data(i)
            super().run()
            llo_points.append(self.get_rmse_llo_index(i))
            #llo_points.append(self.get_rmse_llo_index_val(i))
        #get mean and variance
        mean = np.mean(llo_points)
        var = np.std(llo_points)
        return mean, var

class KernelRegression(SharedFunctions):
    # used in task h

    x = np.empty([1]) # x for plotting
    xdata = np.empty([1]) # x pos data, which will be manipulated
    data = np.empty([1])# y pos data, which will be manipulated
    saveddata = np.empty([1])# y pos data, which won't be manipulated
    xsaved = np.empty([1])# x pos data, which won't be manipulated
    n = 1
    var = 0#variance used to compute cernel

    def __init__(self, n):
        super().__init__()
        self.x = np.arange(0, 6.01, 0.01)
        self.var = 0.15
        self.data = np.array(super().load_data('data/training_data.txt')[1])
        self.xdata = np.array(super().load_data('data/training_data.txt')[0])
        self.saveddata = self.data
        self.xsaved = self.xdata
        self.split_data(n)
        return

    def exp_squared_kernel(self, x1, x2):
        """
        get k(x1, x2) with exp function
        :param x1:
        :param x2:
        :return:
        """
        return math.exp(-1/self.var*np.linalg.norm(x1-x2)**2)

    def kernel(self):
        """
        :return: kernel matrix with n datapoints
        """
        k_mat = np.zeros((self.xdata.size, self.xdata.size))
        for x1 in self.xdata:
            for x2 in self.xdata:
                k_mat[np.where(self.xdata == x1), np.where(self.xdata == x2)] = self.exp_squared_kernel(x1, x2)
        return k_mat

    def kernel_vector(self, x):
        """
        :param x: x value you want to know
        :return: k vector to calc f(x)
        """
        k = []
        for x2 in self.xdata:
            k.append(self.exp_squared_kernel(x, x2))
        return np.array(k)

    def get_y(self, x):
        """
        calculates f(x)
        :param x: value you want to know
        :return: f(x); predicted value
        """
        return self.kernel_vector(x) @ np.linalg.inv(self.kernel()) @ self.data

    def get_y_for_x_vector(self, x_vector):
        """
        :param x_vector: x vector , every element will predicted and written at the same pos in y
        :return: predicted vector for x
        """
        x_v = []
        for x in x_vector:
            x_v.append(self.get_y(x))
        return np.array(x_v)

    def get_data_x(self):
        """
        :return: the used x data
        """
        return self.xdata

    def get_x_saved(self):
        """
        :return: x value of all loaded data points
        """
        return self.xsaved

    def get_data_saved(self):
        """
        :return: data of all loaded data points
        """
        return self.saveddata

    def split_data(self, n):
        """
        splits data into n values
        :param n: model order
        :return:
        """
        if n == 1:
            self.data = self.saveddata[14]
            self.xdata = self.xsaved[14]
        elif 1 < n & n < 30:
            i = 30/(n-1)
            data = []
            x = []
            for v in range(0, 30, int(i)):
                data.append(self.saveddata[v])
                x.append(self.xsaved[v])
            self.data = np.array(data)
            self.xdata = np.array(x)
        else:
            return
        return

    def get_rmse_validation_data(self):
        val_data = self.load_data('data/validation_data.txt')[1]
        val_axis = self.load_data('data/validation_data.txt')[0]

        diff = 0
        for x in val_axis:
            diff += (val_data[np.where(val_axis == x)] - self.get_y(x)) ** 2
        return math.sqrt(1 / val_data.size * int(diff))
























