# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 23:05:00 2017

@author: raghavrastogi
"""

import pandas as pd
df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                      ['red', 'L', 13.5, 'class2'],
                      ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
#print df

size_mapping = {
'XL': 3,
'L': 2,
'M': 1}
df['size'] = df['size'].map(size_mapping)

from sklearn.preprocessing import LabelEncoder
#inv_size_mapping = {v: k for k, v in size_mapping.items()}
import numpy as np
class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)
#inv_class_mapping = {v: k for k, v in class_mapping.items()}
#>>> df['classlabel'] = df['classlabel'].map(inv_class_mapping)
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()
print ohe