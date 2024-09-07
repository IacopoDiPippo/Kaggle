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

# Pulisce tutte le figure
plt.close('all')


train=pd.read_csv("Kaggle1/Accademic_Success_Kaggle/playground-series-s4e6/train.csv", index_col='id')
test=pd.read_csv("Kaggle1/Accademic_Success_Kaggle/playground-series-s4e6/test.csv", index_col='id')

# Target distribution
# Set the figure size and create a count plot
plt.figure(figsize=(10, 8))
ax = sns.countplot(x='Target', data=train, palette='pastel')

# Add labels to each bar in the plot
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 3, f'{int(p.get_height())}', ha="center")


plt.ylabel('Count')
plt.title('Target Distribution')
plt.show()

