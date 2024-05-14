#! /home/schen/final_proj/env/bin/python3.12

import pandas as pd
from process import get_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


x, y = get_data()

n_comp = []
score = []

scaling = StandardScaler()
dat_scaled = scaling.fit_transform(x)

for n in range(2, 12):
    kpca = KernelPCA(n_components=n)
    dat_kpca = kpca.fit_transform(dat_scaled)

    x_train, x_test, y_train, y_test = train_test_split(dat_kpca, y, test_size=.2, random_state=0)


    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)

    n_comp.append(n)
    score.append(acc)


plt.figure()
plt.scatter(n_comp, score)
plt.savefig('./fig/kpca.png')
