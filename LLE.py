#! /home/schen/final_proj/env/bin/python3.12

from process import get_data
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import numpy as np

x, y = get_data()

scaler = StandardScaler()
dat_scaled = scaler.fit_transform(x)

# n_comp = []
# score = []
# for n in range(2, 12):
#     embedd = LocallyLinearEmbedding(n_components = n)
#     dat_embedd = pd.DataFrame(embedd.fit_transform(dat_scaled))

#     x_train, x_test, y_train, y_test = train_test_split(dat_embedd, y, test_size=.2, random_state=0)

#     clf = KNeighborsClassifier(n_neighbors=2)
#     clf.fit(x_train, y_train)
#     acc = clf.score(x_test, y_test)

#     n_comp.append(n)
#     score.append(acc)


# plt.figure() 
# plt.scatter(n_comp, score)
# plt.title('Components vs Acc Under LLE')
# plt.xlabel('Number of Components')
# plt.ylabel('KNN Accuracy')
# plt.savefig('./fig/LLE.png')

embedd = LocallyLinearEmbedding(n_components=11)
dat_embedd = pd.DataFrame(embedd.fit_transform(dat_scaled))

# score = []

# for n in range(20):
#     x_train, x_test, y_train, y_test = train_test_split(dat_embedd, y, test_size=.2)
    
#     clf = KNeighborsClassifier(n_neighbors=2)
#     clf.fit(x_train, y_train)
#     acc = clf.score(x_test, y_test)
#     score.append(acc)

# print(np.mean(score))


#------------------------------
# Using cross validation

knn_cv = KNeighborsClassifier(n_neighbors = 2)
cv_score = cross_val_score(knn_cv, dat_embedd, y, cv = 10)

print(np.mean(cv_score))
    