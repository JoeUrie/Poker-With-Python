#Term Project - CIS 467
#Author: Joseph Urie
#        jgurie@syr.edu
#Poker With Python

import pandas as pd
import numpy as np
from sklearn import svm

#Preproccessing the data
dataset_url = "C:/Users/Joseph Urie/Documents/CIS 467/Term Project/poker-hand-training-true.csv"
data_train = pd.read_csv(dataset_url, header=None, sep=',')

trainset_url = "C:/Users/Joseph Urie/Documents/CIS 467/Term Project/poker-hand-testing.csv"
data_test = pd.read_csv(trainset_url, header=None, sep=',')

data_data = data_train[[0,1,2,3,4,5,6,7,8,9]]
data_target = data_train[10]

test_data = data_test[[0,1,2,3,4,5,6,7,8,9]]
test_target = data_test[10]

#System Interface
entry = input("Hello, welcome to poker 1.0. Please enter how many entries you would like to test. (Max = 1,000,000)")
entryNum = int(entry)

#Training
print("Training...")
clf = svm.SVC(gamma=0.01, C=100)
clf.fit(data_data, data_target)

#Testing
print("Testing...")
clf.predict(test_data[-(entryNum):])

#Results
correct = 0
myList = np.array(test_target[-(entryNum):])
mySecondList = (clf.predict(test_data[-(entryNum):]))
for x in range(0, (len(np.array(test_data[-(entryNum):])))):
    if (myList[x] == mySecondList[x]):
        correct += 1
print("The machine correctly guessed " + str((correct/entryNum)*100) + "% of the poker hands.")