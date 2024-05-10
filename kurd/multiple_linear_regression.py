import random
import math
import time
import numpy as np
from kurd.base_model import KurdModel


class MultiLinearRegression(KurdModel):
    def __init__(self, data):
        super().__init__(data)
        # Create a slope for each input
        self.slopes = []
        self.intercept = 0
        self.loss = 0
        self.slope_gradients = []
        self.intercept_gradient = 0

    def __call__(self, x):
        return self.forward(x)

    def fit(self, epoch, lr):
        self.slopes = self.estimate_slopes()
        self.intercept = self.estimate_intercept()
        loss_list = np.array([])
        slope_gr_list = []
        intercept_gr_list = np.array([])
        for i in range(epoch):
            for ii, x in enumerate(self.X):
                # Make prediction
                pred = self.forward(x)
                true_mean = np.mean(self.y[ii])

                # Calculate Loss
                loss = self.MSELoss(true_mean, pred)
                loss_list = np.append(loss_list, loss)

                # Calculate gradients
                dm, db = self.calculate_gradient(x, true_mean, pred)
                slope_gr_list.append(dm)
                intercept_gr_list = np.append(intercept_gr_list, db)

            self.loss = np.mean(loss_list)
            self.slope_gradients = self.mean_gradients(slope_gr_list)
            self.intercept_gradient = np.mean(intercept_gr_list)
            self.train(lr)

    # Apply gradient descent
    def train(self, lr):
        for i, s_gr in enumerate(self.slope_gradients):
            self.slopes[i] -= s_gr * lr
        self.intercept -= self.intercept_gradient * lr

    def forward(self, X):
        y = []
        for i, x in enumerate(X):
            y.append(x * self.slopes[i])

        return sum(y) + self.intercept

    def estimate_slopes(self):
        y_mid_point = math.floor(len(self.y)/2)
        x_mid_point = math.floor(len(self.X[0])/2)

        # f/s h = first/second half
        y_fh_mean = 0
        y_sh_mean = 0
        for i, yy in enumerate(self.y):
            if i < y_mid_point:
                y_fh_mean += np.mean(yy)
            else:
                y_sh_mean += np.mean(yy)

        y_fh_mean /= y_mid_point
        y_sh_mean /= y_mid_point if len(self.y) % 2 == 0 else (y_mid_point + 1)

        slopes = []

        for i in range(len(self.X[0])):
            x_list = np.array([])
            for j in range(len(self.X)):
                x_list = np.append(x_list, self.X[j][i])
            x_fh_mean = np.mean(x_list[:x_mid_point])
            x_sh_mean = np.mean(x_list[x_mid_point:])
            slopes.append((y_fh_mean - y_sh_mean) / (x_fh_mean - x_sh_mean))

        return slopes

    def estimate_intercept(self):
        x_means = []
        for i in range(len(self.X[0])):
            x_list = np.array([])
            for j in range(len(self.X)):
                x_list = np.append(x_list, self.X[j][i])
            x_means.append(np.mean(x_list))

        x_slope_sum = 0
        for i, xm in enumerate(x_means):
            x_slope_sum += self.slopes[i] * xm

        y_mean = 0
        for yy in self.y:
            y_mean += np.mean(yy)

        y_mean /= len(self.y)

        intercept = -x_slope_sum + y_mean
        return intercept

    def MSELoss(self, true, prediction):
        return (prediction - true) ** 2

    def calculate_gradient(self, inputs, true, prediction):
        slope_gradient_list = []
        for x in inputs:
            # gradient of MSE with respect to slope
            slope_gradient_list.append(2 * (prediction - true) * x)

        # gradient of MSE with respect to intercept (b)
        db = 2 * (prediction - true)

        return (slope_gradient_list, db)

    def mean_gradients(self, gr_list):
        mean_gr = []
        gr_sum = []
        # accumulate all gradients
        for i in range(len(gr_list[0])):
            curr_gr = []
            for j in range(len(gr_list)):
                curr_gr.append(gr_list[j][i])
            gr_sum.append(curr_gr)

        # calculate the mean for each gradient
        for gr in gr_sum:
            mean_gr.append(sum(gr) / len(gr))

        return mean_gr
