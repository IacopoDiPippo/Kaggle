import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from model import Model
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier


#print(os.getcwd())
#os.chdir('C:/Users/iacop/Documents/GitHub/Kaggle1/Accademic_Success_Kaggle')



train=pd.read_csv("Kaggle1/Accademic_Success_Kaggle/playground-series-s4e6/train.csv")
test=pd.read_csv("Kaggle1/Accademic_Success_Kaggle/playground-series-s4e6/test.csv")


Id_test=test['id']

#print(train.head())
#print(test.head())

#print(train.info())

label=train['Target']
train=train.drop('Target',axis=1)

#print(label)
#print(train.head())
#print(train)


label=label.map({'Graduate': 1, 'Dropout': 0,'Enrolled':2})

#print(label)
#print(train.head())

print(train.shape[1])


model=Model()

#loss=nn.CrossEntropyLoss()

#optim=optim.Adam(lr=0.01, weight_decay=0.01)



model2=DecisionTreeClassifier()


model2.fit(train,label)


print('Model_score:',model2.score(train,label))

predictions=model2.predict(test)

output=pd.DataFrame({'id':Id_test, 'Target': predictions})

output=output.replace({1: 'Graduate', 0: 'Dropout',2:'Enrolled'})

os.chdir('C:/Users/iacop/Documents/GitHub/Kaggle1/Accademic_Success_Kaggle')

output.to_csv('submission.csv', index=False)
