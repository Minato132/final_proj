#! /home/schen/final_proj/env/bin/python3.12

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding

with open('archive/proc_data.csv') as file:
    data = pd.read_csv(file)

y = data['loan_status']

X = data.drop(columns='loan_status')

embedding = LocallyLinearEmbedding(n_components=2)

x_dr = embedding.fit_transform(X)

x_transform = pd.DataFrame(x_dr)

print(x_transform)







