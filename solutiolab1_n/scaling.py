#!/usr/bin/python

import pandas as pd
import numpy as np


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

#header is in row 0
male100 = pd.read_csv('../data/male100.csv',header=0)

scaledMale100 = scaleData(male100,0,'Time')

print scaledMale100


scaledMale100 = scaleData(male100,1,'Time')
print scaledMale100
