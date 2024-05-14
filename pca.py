#! /home/schen/final_proj/env/bin/python3.12

from process import get_data
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

x, y = get_data()


scaling = StandardScaler()
scaling.fit(x)
scaled_data = scaling.transform(x)

components = PCA(n_components=11)    
dat_pca = pd.DataFrame(components.fit_transform(scaled_data))

x_train, x_test, y_train, y_test = train_test_split(dat_pca, y, test_size=.2, random_state=0)

clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(x_train, y_train)
acc = clf.score(x_test, y_test)

print(acc)

# From graph it seems the most optimal amount of components is 7
