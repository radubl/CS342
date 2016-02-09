#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn import linear_model as lm
from sklearn.cross_validation import cross_val_score as cv_score
from sklearn import tree 
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV as gscv

def scaleData(dataFrame,flag):
    df = dataFrame.copy()

    for var in df:
        mean = df[var].mean()
        std = df[var].std()
        l1 = (df[var].abs()).sum()
        l2 =  np.sqrt((df[var]**2).sum())

        if(flag == 1):
            df[var] = (df[var]-mean)/std
        elif (flag == 2):
            df[var] = df[var]/l1
        else :
        	df[var] = df[var]/l2

    return df

def getAbaData(rawAbaData):

	data = {}

	# sepparating the target values and removing them from the dataset
	target = rawAbaData['rings']
	rawAbaData = rawAbaData.drop('rings', axis = 1);

	data['Raw Data'] = rawAbaData

	rawAbaDataCopy = rawAbaData.copy()

	# classifiyng the categorical 'sex' feature - pandas specific for 1-of-K encoder thingy
	classifiedSex = pd.get_dummies(rawAbaDataCopy['sex'])
	rawAbaDataCopy = rawAbaDataCopy.drop('sex', axis = 1);

	oneHotAbaDataCopy = rawAbaDataCopy.copy()

	rawAbaDataCopy = scaleData(rawAbaDataCopy,1)

	oneHotAbaDataCopy= scaleData(oneHotAbaDataCopy,0)

	meanScaledOneHot = pd.concat([classifiedSex,rawAbaDataCopy], axis=1)

	matrixScaledOneHot = pd.concat([classifiedSex,oneHotAbaDataCopy], axis=1)

	data['meanScaledOneHot'] = meanScaledOneHot
	data['matrixScaledOneHot'] = matrixScaledOneHot

	return (data,target)
	
def regressAba(data):

	datasetVariants = getAbaData(data)
	target = datasetVariants[1]
	datasetVariants = datasetVariants[0]

	exhaustiveCVPipeline = {
		'OLS' : {
			'model' : lm.LinearRegression(),
			'parameters' : {}
		},
		'Ridge' : {
			'model' : lm.Ridge(),
			'parameters' :  {'alpha' : np.arange(0.05, 1, 0.05)}
		# },
		# 'Lasso' : {
		# 	'model' : lm.Lasso(),
		# 	'parameters' :  {'alpha' : np.arange(0.05, 1, 0.05)}
		# },
		# 'k-NN' : {
		# 	'model' : neighbors.KNeighborsRegressor(),
		# 	'parameters' :  {'weights' : ['uniform','distance'],
		# 					 'leaf_size' : np.arange(5, 100, 1),
		# 					 'n_neighbors' : np.arange(3, 100, 1)}
		# },
		# 'D-Trees' : {
		# 	'model' : tree.DecisionTreeRegressor(),
		# 	'parameters' :  {'max_depth' : np.arange(5, 100, 1)}
		}}

	print 'Starting Exhausting Regression Search on Abalone Data:\n'

	for description,data in datasetVariants.items():
		print '\nUsing dataset: ' + description + '\n'

		for modelName,attributes in exhaustiveCVPipeline.items():

			gscv_instance = gscv(attributes['model'], attributes['parameters'], cv = 10)

			copy = data.copy()

			gscv_instance.fit(copy,target)

			print modelName, gscv_instance.best_score_, gscv_instance.best_params_

def classifyDiabetes(data):

	target = data['class']

	data = data.drop('class', axis = 1);

	data = scaleData(data)

	models = [
		('k-NN', neighbors.KNeighborsClassifier(n_neighbors=5)),
		('D-Trees', tree.DecisionTreeClassifier()),]


	print '\nClassification on Diabetes Data:\n'

	for model_tuple in models:
		copy = data.copy()
		scores = cv_score(model_tuple[1],copy,target,cv = 10)
		print model_tuple[0], np.mean(scores), np.std(scores)

abalone = pd.read_csv('abalone.csv', header=0)
diabetes = pd.read_csv('diabetes.csv', header=0)

regressAba(abalone)

# classifyDiabetes(diabetes)