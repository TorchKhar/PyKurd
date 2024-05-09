import math
import numpy as np
from kurd.base_model import KurdModel


class LinearRegression(KurdModel):
    def __init__(self, data):
        super().__init__(data)
        self.slope = 0
        self.intercept = 0
        self.loss = 0
        self.slope_gradient = 0
        self.intercept_gradient = 0

    def __call__(self, x):
        return self.forward(x)

    def fit(self, epoch, lr):
        # estimate the slope and intercept
        self.slope = self.estimate_slope()
        self.intercept = self.estimate_intercept()

        loss_list = np.array([])
        slope_gr_list = np.array([])
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
                slope_gr_list = np.append(slope_gr_list, dm)
                intercept_gr_list = np.append(intercept_gr_list, db)

            self.loss = np.mean(loss_list)
            self.slope_gradient = np.mean(slope_gr_list)
            self.intercept_gradient = np.mean(intercept_gr_list)
            self.train(lr)

    # Apply gradient descent
    def train(self, lr):
        self.slope -= self.slope_gradient * lr
        self.intercept -= self.intercept_gradient * lr

    def estimate_slope(self):
        x_mid_point = math.floor(len(self.X) / 2)
        first_half_x = self.X[:x_mid_point]
        second_half_x = self.X[x_mid_point:]

        first_half_y = np.array([])
        second_half_y = np.array([])

        for i in range(len(self.y)):
            if i < x_mid_point:
                first_half_y = np.append(first_half_y, np.mean(self.y[i]))
            else:
                second_half_y = np.append(second_half_y, np.mean(self.y[i]))

        slope = (np.mean(second_half_y) - np.mean(first_half_y)) / \
            (np.mean(second_half_x) - np.mean(first_half_x))
        return slope

    def estimate_intercept(self):
        mean_x = np.mean(self.X)
        each_y_mean = np.array([])
        for i in range(len(self.y)):
            each_y_mean = np.append(each_y_mean, np.mean(self.y[i]))

        mean_y = np.mean(each_y_mean)
        mean_y
        intercept = -(self.slope * mean_x) + mean_y
        return intercept

    def forward(self, x):
        return x * self.slope + self.intercept

    def MSELoss(self, true, prediction):
        return (prediction - true) ** 2

    def calculate_gradient(self, input, true, prediction):
        # gradient of MSE with respect to slope (m)
        dm = 2 * (prediction - true) * input
        # gradient of MSE with respect to intercept (b)
        db = 2 * (prediction - true)

        return (dm, db)
