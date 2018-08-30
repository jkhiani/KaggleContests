import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import scipy
import codecs

#read in data
trainData = pd.read_json(codecs.open('train.json','r','utf-8'))
testData = pd.read_json(codecs.open('test.json', 'r', 'utf-8'))


#generate a list of all unique ingredients
masterIngredients = []

for item in trainData['ingredients']:
    for ingredient in item:
        if ingredient not in masterIngredients:
            masterIngredients.append(ingredient)


#one hot encode every set of ingredients in the training data
trainList = []
for item in trainData['ingredients']:
    newList = []
    for x in range(len(masterIngredients)):
        newList.append(0)

    for ingredient in item:
        for x in range(len(masterIngredients)):
            if masterIngredients[x] == ingredient:
                newList[x] = 1
                break

    trainList.append(newList)


#one hot encode every set of ingredients in the test data
testList = []
for item in testData['ingredients']:
    newList = []
    for x in range(len(masterIngredients)):
        newList.append(0)

    for ingredient in item:
        for x in range(len(masterIngredients)):
            if masterIngredients[x] == ingredient:
                newList[x] = 1
                break

    testList.append(newList)


#convert to numpy arrays
X = np.array(trainList)
y = np.array(trainData['cuisine'])

#fit
clf = MultinomialNB()
clf.fit(X,y)



'''
Testing accuracy based on train data
'''
#predictions = clf.predict(trainList)
#print(clf.score(trainList, trainData['cuisine']))


'''
Making predictions based on test data
'''
# predictions = clf.predict(testList)

# for x in range (len(predictions)):
#     print("Prediction: ", predictions[x])
#     print("Ingredients: ", testData['ingredients'][x])
#     print()
#     print()




