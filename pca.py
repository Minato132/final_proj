#! /home/schen/final_proj/env/bin/python3.12

from process import get_data
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x, y = get_data()


scaling = StandardScaler()
scaling.fit(x)
scaled_data = scaling.transform(x)

# components = PCA(n_components=11)    
# dat_pca = pd.DataFrame(components.fit_transform(scaled_data))

# x_train, x_test, y_train, y_test = train_test_split(dat_pca, y, test_size=.2, random_state=0)

# clf = KNeighborsClassifier(n_neighbors=2)
# clf.fit(x_train, y_train)
# acc = clf.score(x_test, y_test)

# print(components.explained_variance_ratio_)


# From graph it seems the most optimal amount of components is 7
n_comp = []
score = []
for n in range(2, 12):
    components = PCA(n_components=n)
    dat_pca = pd.DataFrame(components.fit_transform(scaled_data))

    x_train, x_test, y_train, y_test = train_test_split(dat_pca, y, test_size=.2)
    
    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    
    n_comp.append(n)
    score.append(acc)

plt.figure()
plt.scatter(n_comp, score)
plt.title('Components vs. Acc Under PCA')
plt.xlabel('Number of Components')
plt.ylabel('KNN Accuracy')
plt.savefig('./fig/pca.png')
