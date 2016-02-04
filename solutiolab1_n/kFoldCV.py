#!/usr/bin/python

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pandas.stats.api import ols
from numpy.linalg import inv as inverse
from scipy import stats
from sklearn import cross_validation as cv

def matrixLinReg(dataFrame,xLabel,tLabel,order):
    df = dataFrame.copy() # copy dataframe so don't alter original dataset
    df['1s'] = 1 

    x = df.as_matrix(columns = ['1s'])
    t = df.as_matrix(columns = [tLabel])

    #for loop to add higher order terms
    for i in range(1,order+1):
        xn = df[xLabel]**i
        x = np.insert(x,i,xn,axis=1)

    #matrix calculations needed
    xT = x.transpose()
    xTx = np.dot(xT,x)
    invxTx = inverse(xTx)
    
    w = np.dot(np.dot(invxTx,xT),t)

    return w

#calculate the root mean squared error
def rmse(testData,tLabel):
    return np.sqrt(((testData['Prediction']-testData[tLabel])**2).mean())
    

#predict time given model and year for instances in the test set
def predictions(dataFrame,xLabel,tLabel,modelWeights,order):
    testData = dataFrame.copy()

    pred = modelWeights.item((0,0))

    for i in range(1,order+1):
        pred += modelWeights.item((i,0))*(testData[xLabel]**i)

    testData['Prediction'] = pred
    
    return rmse(testData,tLabel)


def crossValidation(k,dataFrame,order):
    #kf contains indices of instances in each fold - use KFold to split
    kf = cv.KFold(len(dataFrame),k)
    sum_RMSE = 0

    #KFold returns a list containing the training instances and testing instances in each list item
    #will have 10 items in the list if using 10-fold CV
    for train, test in kf:
        trainingDF = dataFrame.iloc[train].copy()
        testingDF = dataFrame.iloc[test].copy()
        
        #create model using the training data only
        model_w = matrixLinReg(trainingDF,'Year','Time',order)
        #calculate the RMSE of each model when testing on the holdout set
        sum_RMSE += predictions(testingDF,'Year','Time',model_w,order)

    return sum_RMSE/k

#------read in data ----------
male100 = pd.read_csv('../data/male100.csv', header=0)
#cross validation with 5 folds for 1st order LinReg
error = `(5,male100,1)

print error

#cross validation with 5 folds for 3rd order LinReg
error = crossValidation(5,male100,3)

print error
