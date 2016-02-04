#!/usr/bin/python

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pandas.stats.api import ols
from numpy.linalg import inv as inverse
from scipy import stats

# f(x;w0,w1) = w0 + w1*x
def matrixLinReg(dataFrame,xLabel,tLabel,order):
    df = dataFrame.copy() # copy dataframe so don't alter original dataset
    
    #need to add 1s to matrix for intercept calculations - can do in the data frame or in the matrix
    df['1s'] = 1 

    #note that 1s are at the start - will effect which item of matrix w is the intercept
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


#linear regression - non-matrix version - harder to scale for higher order terms
def myLinReg(dataFrame,xLabel,tLabel):
    #need to copy the entire dataframe - otherwise treats as reference so columns xt and 
    #x2 are added to the original male100 dataframe
    df = dataFrame.copy()
    x = df[xLabel]
    t = df[tLabel]
    df['xt'] = x*t
    df['x2'] = x*x


    avgx = df[xLabel].mean()
    avgt = df[tLabel].mean()
    avgxt = df['xt'].mean()
    avgx2 = df['x2'].mean()

    
    w1 = (avgxt - (avgx*avgt))/(avgx2-(avgx*avgx))
    w0 = avgt - (w1*avgx)
    
    return w0,w1

#predicting for y = w0 + w1x, would need to alter if including higher order terms 
def predict(x,w0,w1):
    return w0 + (w1 * x)


def scaleData(dataFrame,flag,tLabel):
    df = dataFrame.copy()

    for var in df.drop(tLabel,axis=1):
        mean = df[var].mean()
        std = df[var].std()
        l1 = (df[var].abs()).sum()

        if(flag == 1):
            df[var] = (df[var]-mean)/std
        else:
            df[var] = df[var]/l1

    return df

#------read in data ----------
male100 = pd.read_csv('../data/male100.csv', header=0)

print "Slope & intercept using scipy linregress"
slope, intercept, r_val, p_val, std_err = stats.linregress(male100['Year'],male100['Time'])
print slope, intercept


#use own OLS function
print "Intercept & slope own implementation"
w0,w1 = myLinReg(male100, 'Year','Time')
print w0,w1


#OLS on scaled data
print "Intercept & slope using scaled data"
scaledMale100 = scaleData(male100,1,'Time')

w2,w3 = myLinReg(scaledMale100, 'Year','Time')
print w2,w3

#shows that different parameters are because on different scale
y = w2 + w3*scaledMale100['Year']
scaledMale100.plot(x=0,y=1,kind='scatter')
plt.plot(scaledMale100['Year'],y,'r-',color = 'r')
plt.show()

#make prediction of time in 2025
print "Prediction of time in 2025"
print predict(2025,w0,w1)

#using matrix equation for OLS - easier to include higher order terms
print "Parameters of lin reg using matrix equ. and higher order terms" 
w = matrixLinReg(male100, 'Year', 'Time', 3)

for i in range(0,w.size):
    print w.item((i,0))

#get w weights from the returned matrix
cubicW0 = w.item((0,0))
cubicW1 = w.item((1,0))
cubicW2 = w.item((2,0))
cubicW3 = w.item((3,0))

year = male100['Year']

#plotting data points
male100.plot(x=0,y=1,kind='scatter')

#equations of linear regression lines
y = w0 + w1*male100['Year']
y2 = cubicW0 + cubicW1*year + cubicW2*(year**2) + cubicW3*(year**3)


plt.plot(male100['Year'],y,'r-',color = 'r')
plt.plot(male100['Year'],y2,'r-',color = 'g')
plt.show()
