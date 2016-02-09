
# coding: utf-8

from sklearn.cluster import KMeans
import numpy as np
import csv

# Read data set
data = []
with open('../dataSets/sampleData.csv', 'rU') as csvfile:
    reader  = csv.reader(csvfile)
    for line in reader:
        data.append([float(line[0]), float(line[1])])
x_train = np.array(data)
x_test = np.array([[26,17] , [6.74,10.45], [25., 15.]])

# n_clusters = no. of clusters in which data set is to be divided
model = KMeans(n_clusters=2, random_state=0)
model.fit(x_train)
predicted= model.predict(x_test)
print predicted
print model.cluster_centers_
