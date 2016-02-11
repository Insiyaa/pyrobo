from sklearn.ensemble import RandomForestClassifier
import numpy as np


import csv

# Read Data Set
data = []
with open('../dataSets/sampleData.csv', 'rU') as csvfile:
    reader  = csv.reader(csvfile)
    for line in reader:
        data.append([float(line[0]), float(line[1])])
x_train = np.array(data)
y_train = [2,1,0,2,0,1,2,0,0,0,0,0,0,1,0,0,0,0,0]
x_test = np.array([[26,17] , [6.74, 10.45], [25., 15.]])
model= RandomForestClassifier(n_estimators = 100)
model.fit(x_train, y_train)
predicted= model.predict(x_test)
print predicted

