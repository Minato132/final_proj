#! /home/schen/final_proj/env/bin/python3.12

from process import get_data
import pandas as pd
from sklearn.manifold import SpectralEmbedding
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

x, y = get_data()

n_comp = []
score = []

for n in range(2, 12):
    embedd = SpectralEmbedding(n_components=n)
    dat_embedd = pd.DataFrame(embedd.fit_transform(x))
    
    x_train, x_test, y_train, y_test = train_test_split(dat_embedd, y, test_size=.2)
    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(dat_embedd
    