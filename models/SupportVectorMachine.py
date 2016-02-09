
# coding: utf-8

from sklearn import svm
import numpy as np
import csv

# Read data set
data = []
with open('../dataSets/sampleData.csv', 'rU') as csvfile:
    reader  = csv.reader(csvfile)
    for line in reader:
        data.append([float(line[0]), float(line[1])])
x_train = np.array(data)
y_train = [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
x_test = np.array([[26,17] , [6.74,10.45], [25., 15.]])

model = svm.SVC()
model.fit(x_train, y_train)
model.score(x_train, y_train)
predicted= model.predict(x_test)
print predicted
