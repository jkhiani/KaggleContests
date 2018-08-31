import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

trainData = pd.read_csv("train.csv")
testData = pd.read_csv("test.csv")

y = np.array(trainData['Cover_Type'])

trainData.drop(['Cover_Type'], axis=1, inplace=True)

X = np.array(trainData)

clf = GaussianNB()
clf.fit(X,y)

'''
Testing accuracy based on train data
'''
predictions = clf.predict(X)
print(clf.score(X, y))


'''
Making predictions based on test data
'''
predictions = clf.predict(testData)

print(predictions)




