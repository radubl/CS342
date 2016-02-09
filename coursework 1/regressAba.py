#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn import linear_model as lm
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsRegressor as knnRegressor
from sklearn.tree import DecisionTreeRegressor as treeRegressor
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

def printToFile(results):
	with open('./results.txt', 'a')as f1:
		f1.write('=================== Regression ====================\n')
		f1.write('timestamp : ' + str(datetime.datetime.now()) + '\n\n\n\n')
		col_width = max(len(word) for row in results for word in row) + 2
		for row in results:
			text = "".join(word.ljust(col_width) for word in row)
			print text
			f1.write(text + '\n')
		f1.write('\n')

def regressAba(data):

	target = data[8]

	data = data.drop(8, axis = 1)

	data = data.replace(to_replace = 'M', value = -5)
	data = data.replace(to_replace = 'I', value = 0)
	data = data.replace(to_replace = 'F', value = 5)

	classifiedSex = pd.get_dummies(data[0])
	data = data.drop(0, axis = 1);

	data = scaleData(data, 0)

	l1Scaled = scaleData(data, 1)

	meanVarianceScaled = pd.concat([classifiedSex,data], axis=1)
	l1Scaled = pd.concat([classifiedSex,l1Scaled], axis=1)


	models ={
			# model name : (model object, dataset)
			'LinearRegression' : (lm.LinearRegression(), meanVarianceScaled),
			'Ridge' : (lm.Ridge(alpha=0.95), meanVarianceScaled),
			'Lasso' : (lm.Lasso(alpha=0.05), meanVarianceScaled),
			'k-NN' : (knnRegressor(n_neighbors=15, algorithm = 'kd_tree'), meanVarianceScaled),
			'dec-Trees' : (treeRegressor(max_depth=5), l1Scaled)
			}

	results = [['model','mean','std']]

	for modelName,modelTuple in models.items():

		print "Running", modelName

		cv = cross_validation.ShuffleSplit(len(target), n_iter=10)

		scores = cross_validation.cross_val_score(modelTuple[0], modelTuple[1], target, cv=cv)

		results.append([modelName, str(np.mean(scores)), str(np.std(scores))])


	printToFile(results)

abaloneData = pd.read_csv('Abalone/Dataset.data',sep='\s+', header=None)
regressAba(abaloneData)