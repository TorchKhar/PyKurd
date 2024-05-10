import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from kurd.multiple_linear_regression import MultiLinearRegression

X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [5, 6, 7]]
SLOPES = [3, 2, -1]
INTERCEPT = 1.5

y = []
for x in X:
    yi = []
    for i, xx in enumerate(x):
        yi.append(xx * SLOPES[i])
    y.append(np.array([sum(yi) + INTERCEPT]))

model = MultiLinearRegression((X, y))
model.fit(1000, 0.002)

yp = []
for x in X:
    yp.append(model(x))

print(y)
print(yp)
print(model.slopes)
print(model.loss)
