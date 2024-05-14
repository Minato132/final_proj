#! /home/schen/final_proj/env/bin/python3.12

import pandas as pd
from process import get_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

x, y = get_data()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)

clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(x_train, y_train)
acc = clf.score(x_test, y_test)

print(acc)

# Accuracy with raw data is about 60%

