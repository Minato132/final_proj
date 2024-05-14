#! /home/schen/final_proj/env/bin/python3.12

from process import get_data
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x, y = get_data()

scaler = StandardScaler()
dat_scaled = scaler.fit_transform(x)

n_comp = []
score = []
for n in range(2, 12):
    embedd = LocallyLinearEmbedding(n_components = n)
    dat_embedd = pd.DataFrame(embedd.fit_transform(x))

    x_train, x_test, y_train, y_test = train_test_split(dat_embedd, y, test_size=.2, random_state=0)

    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)

    n_comp.append(n)
    score.append(acc)


plt.figure() 
plt.scatter(n_comp, score)
plt.savefig('./fig/LLE.png')
