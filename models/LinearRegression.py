
# coding: utf-8

from sklearn import linear_model
import numpy as np
import scipy as sp

"""
    New version of sklearn requires you to have data in the form of (no_rows, no_colums)
    If your dataset is 1D i.e has only one feature then reshape it with .reshape(-1, 1)
    If your dataset has only one data point then reshape it with .reshape(1, -1)
    This example is to predecit linear equation y = 2x
"""
x_train = np.array([1,1.5, 2, 2.6, 4, 7, 6.1]).reshape(-1,1)
y_target = np.array([2, 3.1, 4, 5, 7.5, 13.66, 12]).reshape(-1,1)
x_test = np.array([3 , 4.5, -5 , -7.11]).reshape(-1,1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_target)
linear.score(x_train, y_target)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
predicted= linear.predict(x_test)
print predicted


