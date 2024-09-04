import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier


print(os.getcwd())
os.chdir('C:/Users/iacop/Documents/GitHub/Kaggle1/House_Kaggle')


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

#print(dataset,label)


model=LinearRegression()

model.fit(dataset, label)

print('Model_score:', model.score(dataset,label))

predictions=model.predict(datatest)

output=pd.DataFrame({'Id':Id_test, 'SalePrice': predictions})

output.to_csv('submission.csv', index=False)



model1=DecisionTreeClassifier()

model1.fit(dataset, label)

print('Model_score:', model1.score(dataset,label))

predictions1=model1.predict(datatest)

output1=pd.DataFrame({'Id':Id_test, 'SalePrice': predictions1})

output1.to_csv('submission1.csv', index=False)





