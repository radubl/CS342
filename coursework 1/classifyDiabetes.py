#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.cross_validation import cross_val_predict as cv_predict
from sklearn.cross_validation import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier as knnClassifier
from sklearn.tree import DecisionTreeClassifier as treeClassifier
from sklearn.preprocessing import Imputer
import datetime

def scaleData(dataFrame,flag):
	df = dataFrame.copy()

	for var in df:
		mean = df[var].mean()
		std = df[var].std()
		l1 = (df[var].abs()).sum()
		l2 =  np.sqrt((df[var]**2).sum())

		if(flag == 0):
			df[var] = (df[var]-mean)/std
		elif (flag == 1):
			df[var] = df[var]/l1
		else :
			df[var] = df[var]/l2

	return df

def dealWithMissingValues(data):
	imputer = Imputer(missing_values=0, strategy="mean", axis=0)
	return imputer.fit_transform(data)

def printToFile(results):
	print ''
	with open('./results.txt', 'a')as f1:
		f1.write('=================== Classification ====================\n')
		f1.write('timestamp : ' + str(datetime.datetime.now()) + '\n\n')
		col_width = max(len(word) for row in results for word in row) + 2

		for row in results:
			text = "".join(word.ljust(col_width) for word in row)
			print text
			f1.write(text + '\n')
		f1.write('\n')

def classifyDiabetes(data):

	target = data[8]

	data = data.drop(8, axis = 1)

	data = dealWithMissingValues(data)

	meanVarianceScaled = scaleData(data, 0)
	l1Scaled = scaleData(data, 1)

	models ={
			# model name : (model object, dataset)
			'k-NN' : (knnClassifier(n_neighbors=5, algorithm = 'kd_tree'), meanVarianceScaled),
			'dec-Trees' : (treeClassifier(max_depth=5), l1Scaled)
			}

	results = [['model','mean','std']]

	for modelName,modelTuple in models.items():

		print "Running", modelName

		shuffled = ShuffleSplit(len(target), n_iter=10)

		data = modelTuple[1]
		model = modelTuple[0]

		scores = []

		for train_index, test_index in shuffled:

			trainSet = data.ix[train_index] 
			trainTarget = target.ix[train_index] 
			testSet = data.ix[test_index] 
			testTarget = target.ix[test_index] 
			model.fit(trainSet,trainTarget)

			predictions = model.predict(testSet)
			scores.append(f1_score(testTarget, predictions))

		results.append([modelName, str(np.mean(scores)), str(np.std(scores))])

	printToFile(results)

diabetesData = pd.read_csv('diabetes.data', header=None)
classifyDiabetes(diabetesData)
