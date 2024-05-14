#! /home/schen/final_proj/env/bin/python3.12

from process import get_data
import pandas as pd
from sklearn.manifold import SpectralEmbedding
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

x, y = get_data()

n_comp = []
score = []

scaling = StandardScaler()
dat_scaled = scaling.fit_transform(x)


# for n in range(2, 12):
#     embedd = SpectralEmbedding(n_components=n)
#     dat_embedd = pd.DataFrame(embedd.fit_transform(dat_scaled))
    
#     x_train, x_test, y_train, y_test = train_test_split(dat_embedd, y, test_size=.2)
#     clf = KNeighborsClassifier(n_neighbors=2)
#     clf.fit(x_train, y_train)
#     acc = clf.score(x_test, y_test)
#     n_comp.append(n)
#     score.append(acc)

# plt.figure()
# plt.scatter(n_comp, score)
# plt.savefig('./fig/eigen.png')

# We found that 7 is a good number of components for spectralembedding thus we we will continue with this
# number and account for the random factors

#-----------------------------------------------------------------------------------

embedd = SpectralEmbedding(n_components=7)
dat_embedd = pd.DataFrame(embedd.fit_transform(dat_scaled))

score = []
for n in range(20):
    x_train, x_test, y_train, y_test = train_test_split(dat_embedd, y, test_size=.2)
    
    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)

    score.append(acc)

print(np.mean(score))

# We get an avg accuracy of .8724