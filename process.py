#! /home/schen/final_proj/env/bin/python3.12

import pandas as pd

def get_data():
    with open('archive/proc_data.csv') as file:
        data = pd.read_csv(file)

    y = data['loan_status']

    X = data.drop(columns='loan_status')

    return (X, y)






