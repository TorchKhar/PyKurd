import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import random
from kurd.linear_regression import LinearRegression

SLOPE = 1.5
INTERCEPT = 2

# Make dummy data
X = [1, 2, 3, 4, 6, 13]
y = []

for i, x in enumerate(X):
  y_for_x = x * SLOPE + INTERCEPT
  y.append(np.array([y_for_x]))
  # add some randomness
  if random.randint(0, 2) == 1:
    y[i] = np.append(y[i], y_for_x + random.randint(-1, 1))

print(X)
print(y)

# Make the model
model = LinearRegression((X, y))
model.fit(200, 0.0001)

print("Performance on train data: ")
for i, x in enumerate(X):
  pred = model(x)

  print(f"Input:{x}  |  Prediction:{pred}  |  True Value:{np.mean(y[i])}")

print(f"Loss: {model.loss}")
print("--------------------------------------------")

# Make some data for test
X_test = [5, 7, 8, 9, 10, 11]
y_test = []

for x in X_test:
  y_for_x = x * SLOPE + INTERCEPT
  y_test.append(np.array([y_for_x]))

print("Performance on test data: ")
for i, x in enumerate(X_test):
  pred = model(x)

  print(f"Input:{x}  |  Prediction:{pred}  |  True Value:{np.mean(y_test[i])}")
