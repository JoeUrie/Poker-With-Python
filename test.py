#Test for Term_Project
#Author: Joseph Urie

import pandas as pd
import numpy as np
from sklearn import svm

#digits = datasets.load_digits()
dataset_url = "C:/Users/Joseph Urie/Documents/CIS 467/Term Project/poker-hand-training-true.csv"
data_train = pd.read_csv(dataset_url, header=None, sep=',')

trainset_url = "C:/Users/Joseph Urie/Documents/CIS 467/Term Project/poker-hand-testing.csv"
data_test = pd.read_csv(trainset_url, header=None, sep=',')

print(data_train.shape)
print(data_test.shape)

data_data = data_train[[0,1,2,3,4,5,6,7,8,9]]
data_target = data_train[10]

test_data = data_test[[0,1,2,3,4,5,6,7,8,9]]
test_target = data_test[10]

clf = svm.SVC(gamma=0.01, C=100)
clf.fit(data_data, data_target)
#print(np.array(test_target[-10:]))
clf.predict(test_data)

correct = 0
myList = np.array(test_target)
mySecondList = (clf.predict(test_data))
for x in range(0, (len(np.array(test_data)))):
    if (myList[x] == mySecondList[x]):
        correct += 1
print(correct)
    
    