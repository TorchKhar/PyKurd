import pandas
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from kurd.multiple_linear_regression import MultiLinearRegression
from sklearn.model_selection import train_test_split

data = pandas.read_csv("./examples/student-mat.csv", sep=";")
data = data[["studytime", "G1", "G2", "G3"]]

X = np.array(data.drop(["G3"], axis=1))
y = np.array(data["G3"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultiLinearRegression((X_train, y_train))
model.fit(2000, 0.001)
print(f"LOSS: {model.loss}")

for i, x_pred in enumerate(X_test):
  y_pred = model(x_pred)
  print(f"PREDICTION: {y_pred} | TRUE: {y_test[i]}")
