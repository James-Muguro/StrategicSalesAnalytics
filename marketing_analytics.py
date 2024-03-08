# To start working with the dataset, we will import the following libraries

import pandas as pd  # Pandas for data manipulation and analysis
import numpy as np   # NumPy for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for basic data visualization
import seaborn as sns  # Seaborn for advanced data visualization
from sklearn.cluster import KMeans  # KMeans for clustering
from sklearn.preprocessing import StandardScaler  # StandardScaler for feature scaling
from sklearn.model_selection import train_test_split  # Train-test split for model evaluation
import xgboost as xgb
import ipywidgets as widgets
from IPython.display import display, clear_output
from tkinter import Tk, filedialog
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, r2_score


# Loading the dataset
def select_file(b):
    clear_output()
    root = Tk()
    root.withdraw()  # Hide the main window
    root.call('wm', 'attributes', '.', '-topmost', True)  # Raise the root to the top of all windows
    b.files = filedialog.askopenfilename(multiple=False)  # List of selected files
    path = b.files
    global df
    df = pd.read_excel(path)
    print(f'Loaded dataframe from {path}')
    display(df.head())

fileselect = widgets.Button(description="File select")
fileselect.on_click(select_file)

display(fileselect)

df.head()


# Understanding the data
df.dtypes
df.shape


# Data Cleaning
# Missing Values
# Drop rows with missing values
df_cleaned = df.dropna()

# Outliers
# Remove outliers in 'Units' column using z-score
from scipy.stats import zscore
df_cleaned = df[(np.abs(zscore(df['Units'])) < 3)]


# Data Analysis
# Key Metrics

# 1. TrafficChannel Performance

# Group by TrafficChannel and calculate metrics
traffic_channel_metrics = df_cleaned.groupby('TrafficChannel')[['Units', 'UnitPrice']].agg({'Units': 'sum', 'UnitPrice': 'mean'}).reset_index()

# Visualization
plt.figure(figsize=(12, 6))
sns.barplot(x='TrafficChannel', y='Units', data=traffic_channel_metrics, palette='viridis')
plt.title('TrafficChannel Performance - Total Units Sold')
plt.xlabel('TrafficChannel')
plt.ylabel('Total Units Sold')
plt.show()

# 2. Segement Preference:

# Group by Segement and calculate metrics
segement_metrics = df_cleaned.groupby('Segement')[['Units', 'UnitPrice']].agg({'Units': 'sum', 'UnitPrice': 'mean'}).reset_index()

# Visualization
plt.figure(figsize=(12, 6))
sns.barplot(x='Segement', y='Units', data=segement_metrics, palette='muted')
plt.title('Segement Preference - Total Units Sold')
plt.xlabel('Segement')
plt.ylabel('Total Units Sold')
plt.show()

# 3. Device Performance:

# Group by Device and calculate metrics
device_metrics = df.groupby('Device')[['Units', 'UnitPrice']].agg({'Units': 'sum', 'UnitPrice': 'mean'}).reset_index()

# Visualization
plt.figure(figsize=(12, 6))
sns.barplot(x='Device', y='Units', data=device_metrics, palette='pastel')
plt.title('Device Performance - Total Units Sold')
plt.xlabel('Device')
plt.ylabel('Total Units Sold')
plt.show()


# Lets Map the states and add a new column called region 

# Define a dictionary that maps states to regions
state_to_region = {
    'Alabama': 'South', 'Alaska': 'West', 'Arizona': 'West', 'Arkansas': 'South', 
    'California': 'West', 'Colorado': 'West', 'Connecticut': 'Northeast', 
    'Delaware': 'South', 'Florida': 'South', 'Georgia': 'South', 'Hawaii': 'West', 
    'Idaho': 'West', 'Illinois': 'Midwest', 'Indiana': 'Midwest', 'Iowa': 'Midwest', 
    'Kansas': 'Midwest', 'Kentucky': 'South', 'Louisiana': 'South', 'Maine': 'Northeast', 
    'Maryland': 'South', 'Massachusetts': 'Northeast', 'Michigan': 'Midwest', 
    'Minnesota': 'Midwest', 'Mississippi': 'South', 'Missouri': 'Midwest', 
    'Montana': 'West', 'Nebraska': 'Midwest', 'Nevada': 'West', 'New Hampshire': 'Northeast', 
    'New Jersey': 'Northeast', 'New Mexico': 'West', 'New York': 'Northeast', 
    'North Carolina': 'South', 'North Dakota': 'Midwest', 'Ohio': 'Midwest', 
    'Oklahoma': 'South', 'Oregon': 'West', 'Pennsylvania': 'Northeast', 
    'Rhode Island': 'Northeast', 'South Carolina': 'South', 'South Dakota': 'Midwest', 
    'Tennessee': 'South', 'Texas': 'South', 'Utah': 'West', 'Vermont': 'Northeast', 
    'Virginia': 'South', 'Washington': 'West', 'West Virginia': 'South', 
    'Wisconsin': 'Midwest', 'Wyoming': 'West'
}

# Create a new column 'Region' in the dataframe
df['Region'] = df['State'].map(state_to_region)


# 4. Region Performance:

# Group by Region and calculate metrics
region_metrics = df.groupby('Region')[['Units', 'UnitPrice']].agg({'Units': 'sum', 'UnitPrice': 'mean'}).reset_index()

# Visualization
plt.figure(figsize=(12, 6))
sns.barplot(x='Region', y='Units', data=region_metrics, palette='deep')
plt.title('Region Performance - Total Units Sold')
plt.xlabel('Region')
plt.ylabel('Total Units Sold')
plt.show()

# 5. Product Preference:

# Group by ProductName and calculate metrics
product_metrics = df.groupby('ProductName')[['Units', 'UnitPrice']].agg({'Units': 'sum', 'UnitPrice': 'mean'}).reset_index()

# Visualization (top 10 products by units sold)
top_products = product_metrics.nlargest(10, 'Units')
plt.figure(figsize=(12, 6))
sns.barplot(x='Units', y='ProductName', data=top_products, palette='colorblind')
plt.title('Top 10 Product Preferences - Total Units Sold')
plt.xlabel('Total Units Sold')
plt.ylabel('Product Name')
plt.show()

 

