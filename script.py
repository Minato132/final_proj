#! /home/schen/final_proj/env/bin/python3.12

import pandas as pd
from sklearn import preprocessing

with open('archive/data.csv') as file:
    data = pd.read_csv(file)

data.drop(columns='loan_id', inplace = True)

label_encoder = preprocessing.LabelEncoder()

obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])

for x in data.columns:
    data.rename(columns={x:x.strip(' ')}, inplace=True)


data.to_csv('./archive/proc_data.csv', index_label=False)
