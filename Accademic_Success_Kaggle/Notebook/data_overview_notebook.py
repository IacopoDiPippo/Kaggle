import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set the style of matplotlib
#%matplotlib inline
plt.style.use('fivethirtyeight')

train=pd.read_csv("Kaggle1/Accademic_Success_Kaggle/playground-series-s4e6/train.csv", index_col='id')
test=pd.read_csv("Kaggle1/Accademic_Success_Kaggle/playground-series-s4e6/test.csv", index_col='id')


# Check if there are any missing values
train.isna().sum().sort_values(ascending=False)

# Check if there are duplicate rows
train.duplicated().sum()

# View the general information of the training dataset
train.info()

# View the statistical description of training dataset
print(train.describe().T)

# Store the names of feature columns
initial_features = list(test.columns)


# Print the number of unique values for each column
for col in train.columns:
print(f'{col} has {train[col].nunique()} values')


# Classify columns for better visualization
# Categorical columns: if the number of unique values is 8 or fewer
cat_cols = [col for col in train.columns if train[col].nunique() <= 8]
# Numerical columns: if the number of unique values is 9 or more
num_cols = [col for col in train.columns if train[col].nunique() >= 9]