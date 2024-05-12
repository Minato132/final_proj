#! /home/schen/final_proj/env/bin/python3.12

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


with open('archive/proc_data.csv') as file:
    data = pd.read_csv(file)

y = data['loan_status']
X = data.drop(columns='loan_status')

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=4)

K = []
training = []
test = []
scores = {}

for k in range(2, 21):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(x_train, y_train)

    train_score =  clf.score(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    K.append(k)

    training.append(train_score)
    test.append(test_score)
    scores[k] = [train_score, test_score]


plt.figure()
plt.scatter(K, training, color = 'g')
plt.scatter(K, test, color = 'k')
plt.xlabel('Num Neighbors')
plt.ylabel('Score')
plt.savefig('./fig1.png')

#fig 1 shows that about 16 is the correct number of neighbors