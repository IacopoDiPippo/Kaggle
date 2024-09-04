import numpy as np
import pandas as pd
import os
import tensorflow_decision_forests as tfdf
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt


print(os.getcwd())
os.chdir('C:/Users/iacop/Documents/GitHub//Kaggle1/House_Kaggle')


dataset=pd.read_csv('train.csv')
datatest=pd.read_csv('test.csv')

Id_test=datatest['Id']

label=dataset['SalePrice']


dataset=dataset.drop(['SalePrice','Id'],axis=1)
datatest=datatest.drop(['Id'],axis=1)


dataset=dataset.select_dtypes(include=['float64', 'int64'])
datatest=datatest.select_dtypes(include=['float64', 'int64'])


dataset=(dataset-dataset.mean())/dataset.std()
datatest=(datatest-datatest.mean())/datatest.std()



dataset = dataset.fillna(0)
datatest = datatest.fillna(0)

model=tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)

print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)


print(dataset.info())







print(dataset,label)

