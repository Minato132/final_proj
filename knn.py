#! /home/schen/final_proj/env/bin/python3.12

import pandas as pd
from process import get_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

x, y = get_data()

score = []
for n in range(20):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    
    score.append(acc)

print(np.mean(score))

# Accuracy with raw data is about 60%

