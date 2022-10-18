# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 00:32:16 2022

@author: my pc
"""

import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt
with open("traind2.pkl", "rb") as f:
       x_train, y_train = pkl.load(f)
with open("testd2.pkl", "rb") as f1:
       x_test, y_test = pkl.load(f1)
sns.countplot(y_train)
plt.title('Hospital-D Train Data (Total=15000)')
plt.xlabel('class')
plt.show()

sns.countplot(y_test)
plt.title('Hospital-D Test Data (Total=5000)')
plt.xlabel('class')
plt.show()