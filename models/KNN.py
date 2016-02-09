
# coding: utf-8

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import csv

# Read data set
data = []
with open('../dataSets/sampleData.csv', 'rU') as csvfile:
    reader  = csv.reader(csvfile)
    for line in reader:
        data.append([float(line[0]), float(line[1])])
x_train = np.array(data)
y_train = [1,2,1,2,2,2,2,0,0,0,0,0,0,1,0,0,0,0,0]
x_test = np.array([[26,17] , [6.74, 10.45], [25., 15.]])

# n_neighbors is to set no of neighbors to be considered
model = KNeighborsClassifier(n_neighbors=2)
model.fit(x_train, y_train)

predicted = model.predict(x_test)
print predicted
