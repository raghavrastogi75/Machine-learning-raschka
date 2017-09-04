# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 21:50:16 2017

@author: raghavrastogi
"""
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
      
#from sklearn.tree import DecisionTreeClassifier
#tree = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=0)
#tree.fit(X_train, y_train)
#X_combined = np.vstack((X_train, X_test))
#y_combined = np.hstack((y_train, y_test))
#plot_decision_regions(X_combined, y_combined,tree)
#plt.xlabel('petal length [cm]')
#plt.ylabel('petal width [cm]')
#plt.legend(loc='upper left')
#plt.show()

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy',n_estimators=10,random_state=1,n_jobs=2)
forest.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined,forest)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2,metric='minkowski')
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined,knn)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.show()