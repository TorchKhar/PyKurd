import math
import numpy as np
from base_model import KurdModel

class LinearRegression(KurdModel):
  def __init__(self, data):
    super().__init__(data)
    self.slope = 0
    self.intercept = 0

  def fit(self, epoch):
    # estimate the slope
    self.slope = self.estimate_slope()
    self.intercept = self.estimate_intercept()

  def train(self):
    return

  def estimate_slope(self):
    x_mid_point = math.floor(len(self.X))
    first_half_x = self.X[:x_mid_point]
    second_half_x = self.X[x_mid_point:]

    first_half_y = np.array([])
    second_half_y = np.array([])

    for i in range(len(self.y)):
      if i < x_mid_point:
        first_half_y = np.append(first_half_y, np.mean(self.y[i]))
      else:
        second_half_y = np.append(second_half_y, np.mean(self.y[i]))

    slope = (np.mean(second_half_y) - np.mean(first_half_y)) / (np.mean(second_half_x) - np.mean(first_half_x))
    return slope

  def estimate_intercept(self):
    mean_x = np.mean(self.X)
    each_y_mean = np.array([])
    for i in range(len(self.y)):
      each_y_mean = np.append(each_y_mean, np.mean(self.y[i]))

    mean_y = np.mean(each_y_mean)
    mean_y
    intercept = -(slope * mean_x) + mean_y
    return intercept