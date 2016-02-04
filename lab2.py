import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knn

def doRelativeScaling(scaleInput):
	mean = scaleInput.mean()
	std = scaleInput.std()

	scaleInput = (scaleInput - mean) / std

	return scaleInput

# takes the vectors of results
def accuracy(predicted,actual):
	trues = 0;
	for x in range(len(predicted)):
		if predicted[x] == actual[x]:
			trues += 1

	return trues/len(predicted)
			

diabetesData = pd.read_csv("diabetes.csv",header=0);

classValues = diabetesData["class"]

# print accuracy(classValues,classValues)

del diabetesData["class"]

scaledData = doRelativeScaling(diabetesData)

# print scaledData

neigh = knn(n_neighbors=1)

neigh.fit(scaledData,classValues) 

print(neigh.predict([[1.3, 1.6, 1.9,0.7,5,2,1,5]])) 