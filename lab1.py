import pandas as pd
import numpy as np

male100 = pd.read_csv("data/male100.csv",header=0)


def scaleData(X,flag):
	if flag == "standard":
		mean = X["Time"].mean()
		std = X["Time"].std()
		return (X - mean) / std
	else:
		return 1;

def linreg(X,t):
	return X.as_matrix(0)
		

print male100
exit()

