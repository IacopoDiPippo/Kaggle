import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from main import train



print((train['Daytime/evening attendance'].describe()))


# Creating a figure with a custom size
plt.figure(figsize=(16, 20))

# Plotting a histogram with custom bin size and label sizes
train['Daytime/evening attendance'].hist(bins=50)

# Customizing the x and y labels' font sizes
plt.xlabel('Values', fontsize=8)
plt.ylabel('Frequency', fontsize=8)

# Show the plot
plt.show()







