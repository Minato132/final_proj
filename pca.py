#! /home/schen/final_proj/env/bin/python3.12

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

with open('archive/proc_data.csv') as file:
    data = pd.read_csv(file)

y = data['loan_status']
x = data.drop(columns='loan_status')

scaling = StandardScaler()
scaling.fit(x)

scaled_data = scaling.transform(x)

components = PCA(n_components=11)
components.fit(scaled_data)
dat_pca = components.transform(scaled_data)

x_train, x_test, y_train, y_test = train_test_split(dat_pca, y, test_size=.2, random_state=4)

clf = KNeighborsClassifier(n_neighbors=11)
clf.fit(x_train, y_train)
acc = clf.score(x_test, y_test)

print(acc)
