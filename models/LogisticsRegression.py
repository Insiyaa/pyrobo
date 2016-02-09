
# coding: utf-8
from sklearn.linear_model import LogisticRegression
import numpy as np
"""
    New version of sklearn requires you to have data in the form of (no_rows, no_colums)
    If your dataset is 1D i.e has only one feature then reshape it with .reshape(-1, 1)
    If your dataset has only one data point then reshape it with .reshape(1, -1)
    This example is to predecit if given number is whole number or decimal (1 for whole number and  0 for decimal)
"""
x_train = np.array([1 , 54, 1.5, 2, 2.6, 4, 7, 6.1 , -1.5, -5, -2.55, -10, 14 , 1.44 ,5.66 ,100]).reshape(-1,1)
y_target = np.array([1, 1 , 0 , 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0 ,0, 1])
x_test  = np.array([3, -4, 4.3 , -0.6, 5, 5.66 , -8]).reshape(-1,1)

model = LogisticRegression()
model.fit(x_train, y_target)
model.score(x_train, y_target)
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)

# Make estimations for test data
predicted= model.predict(x_test)
print predicted

# Get probablity for each estimation
print model.predict_proba(x_test)
