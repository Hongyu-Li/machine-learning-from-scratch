import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('../data/housing.csv', sep=',')
X = np.asarray(data.iloc[:,:-1])
Y = np.asarray(data.iloc[:,-1])


