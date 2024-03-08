# Strategic Sales Analytics

This project revolves around strategic marketing intelligence, leveraging customer segmentation, traffic channel assessment, cost optimization, and machine learning for campaign predictions. By employing advanced analytics and natural language processing, it provides a comprehensive understanding of the market landscape.

## Table of Contents

1. [Understanding the Data](#understanding-the-data)
2. [Data Cleaning](#data-cleaning)
3. [Key Metrics](#key-metrics)
4. [Key Analytics](#key-analytics)
5. [Advanced Analytics](#advanced-analytics)
6. [Machine Learning Models](#machine-learning-models)
7. [Natural Language Processing](#natural-language-processing)

## Understanding the Data

In this section, we first understand the structure of our data. We check the data types of each column to ensure that they are appropriate for our analysis. We also check the shape of our DataFrame to know how many observations and variables we are working with.

## Data Cleaning

Data cleaning is a crucial step in any data analysis. In this process, we handle missing values and outliers. We drop rows with missing values as they can skew our analysis. We also remove outliers in the 'Units' column using z-score. Outliers can significantly affect averages and disrupt the overall understanding of the data distribution.

## Key Metrics

In the data analysis section, we calculate key metrics and create visualizations for:

### TrafficChannel Performance

![alt text](image.png)

The most successful traffic channel is  Organic Search, leading to the highest number of total units sold, exceeding 700,000. This suggests that users who find the product through organic search results are the most likely to make a purchase. SEM (Search Engine Marketing) and SEO (Search Engine Optimization) are also effective channels, each contributing to around 600,000 units sold, indicating that targeted advertising and optimization for search engine results significantly contribute to sales. SMO (Social Media Optimization), Banner, and Affiliate channels also contribute to sales but to a lesser extent. Mail and Email are the least effective channels, with just above 100,000 and less than 100,000 units sold, respectively. This graph provides a clear comparison of the effectiveness of different traffic channels in terms of units sold. It appears that Organic Search, SEM, and SEO are the most effective channels.

### Segment Preference

![alt text](image-1.png)

The 'Extreme' segment has the highest total units sold among the nine different segments displayed. This suggests that the 'Extreme' segment is the most preferred by customers in terms of units sold.

### Device Performance

![alt text](image-2.png)

Desktop and Mobile Device have the highest total units sold, indicating these are the most preferred devices by customers. The Laptop and Television have the fewest units sold.

### Region Performance

![alt text](image-3.png)

The Midwest region has the highest total units sold, followed by the South, West, and Northeast regions. This indicates that the Midwest is the most preferred region by customers in terms of units sold.

### Product Preference

![alt text](image-4.png)

Maximus UM-54 has the highest total units sold, indicating it is the most preferred product among the top 10 listed. The product “Maximus UC-50” has the fewest units sold among the listed products

## Key Analytics

In this section, we perform key analytics including:

### Descriptive statistics

Descriptive statistics provide a detailed summary of the data

Descriptive Statistics for Numerical Columns:
           ProductID                           Date     CustomerID  \
count  611170.000000                         611170  611170.000000   
mean      517.907147  2017-07-17 19:41:37.563688704  137125.513682   
min       392.000000            2015-01-01 00:00:00       1.000000   
25%       449.000000            2016-05-10 00:00:00   67737.000000   
50%       496.000000            2017-07-11 00:00:00  136285.000000   
75%       585.000000            2018-09-20 00:00:00  205672.000000   
max       691.000000            2019-12-31 00:00:00  282597.000000   
std        81.302167                            NaN   80118.485333   

          CampaignID          Units       UnitCost      UnitPrice  \
count  611170.000000  611170.000000  611170.000000  611170.000000   
mean       10.758753       5.497966      86.925476     158.339428   
min         1.000000       1.000000      16.000000      21.000000   
25%         4.000000       3.000000      56.000000      97.000000   
50%        11.000000       5.000000      79.000000     153.000000   
75%        17.000000       8.000000     123.000000     198.000000   
max        22.000000      10.000000     150.000000     444.000000   
std         7.053343       2.873528      37.996861      77.989529   

             ZipCode  
count  611170.000000  
mean    52495.258444  
min      1001.000000  
25%     31410.000000  
50%     52073.000000  
75%     76258.000000  
max     99950.000000  
std     26598.676693  

Descriptive Statistics for Categorical Columns:
          ProductName Category    Segement    TrafficChannel   Device   State  \
count          611170   611170      611170            611170   611170  611170   
unique            173        5           9                 8        5      49   
top     Maximus UM-54    Urban  Moderation  Organic Search    Desktop   Texas   
freq            63966   510849      334616            145495   198755   56092   

        Region  
count   611170  
unique       4  
top      South  
freq    233695  

### correlation analysis

Correlation analysis helps us understand the relationship between different variable

The correlation between UnitCost and UnitPrice is 0.52

![alt text](image-5.png)

The Desktop device has the highest average units sold, while the Television device has the lowest. This indicates that customers prefer to use Desktop devices over others when purchasing unit

### Time series analysis

Time series analysis helps us understand the trend and seasonality in our data.

![alt text](image-6.png)

There are noticeable peaks and troughs, indicating varying sales performance over time. This suggests that certain months or periods within these years had higher sales compared to others

## Advanced Analytics

In this section, we perform advanced analytics including:

### Customer segmentation

Customer segmentation helps us understand the different groups of customers

![alt text](image-8.png)

There are three distinct clusters of customers. Customers in Cluster 0 tend to purchase items with lower UnitCost and UnitPrice, while customers in Cluster 2 purchase items with higher UnitCost and UnitPrice. Cluster 1 customers fall in between. This indicates different customer segments based on their purchasing patterns in terms of UnitCost and UnitPrice.

### Item-item collaborative filtering

Item-item collaborative filtering is a method of making automatic predictions about the interests of a user by collecting preferences from many users.

ProductName
Maximus UM-43    1.000000
Maximus UM-54    0.100722
Maximus UM-11    0.075382
Maximus UM-75    0.058597
Maximus UM-56    0.055813
Name: Maximus UM-43, dtype: float64

## Machine Learning Models

In this section, we implement machine learning models including:

### XGBoost for regression

Mean Squared Error: 0.3500736145583455
Feature Importance of CustomerID: 3.403451046324335e-05
Feature Importance of CampaignID: 1.2340366083662957e-05
Feature Importance of Units: 0.9999285936355591
Feature Importance of UnitCost: 8.53950041346252e-06
Feature Importance of UnitPrice: 1.6602580217295326e-05
Some predictions:
Prediction: 3.5071473121643066, Actual: 3
Prediction: 5.119030475616455, Actual: 5
Prediction: 6.685296535491943, Actual: 7
Prediction: 9.067241668701172, Actual: 10
Prediction: 6.691070556640625, Actual: 7
Prediction: 3.477186918258667, Actual: 3
Prediction: 3.5147688388824463, Actual: 3
Prediction: 3.513033628463745, Actual: 3
Prediction: 6.670936107635498, Actual: 7
Prediction: 5.101507186889648, Actual: 5

### LSTM for time series analysis

Epoch [1/10], Loss: 0.11193192005157471
Epoch [2/10], Loss: 0.11172807216644287
Epoch [3/10], Loss: 0.1116839125752449
Epoch [4/10], Loss: 0.11167553812265396
Epoch [5/10], Loss: 0.11168033629655838
Epoch [6/10], Loss: 0.11169217526912689
Epoch [7/10], Loss: 0.11170084029436111
Epoch [8/10], Loss: 0.11170634627342224
Epoch [9/10], Loss: 0.11171285063028336
Epoch [10/10], Loss: 0.11171510815620422
Mean Squared Error: 8.313689883504606
Some predictions:
Prediction: [5.4780254], Actual: [2]
Prediction: [5.374848], Actual: [8]
Prediction: [5.423319], Actual: [2]
Prediction: [5.3224454], Actual: [3]
Prediction: [5.180523], Actual: [2]

## Natural Language Processing

In this section, we implement natural language processing techniques including:

### Topic modeling with Latent Dirichlet Allocation (LDA)

[nltk_data]   Unzipping tokenizers\punkt.zip.
(0, '0.250*"maximus" + 0.248*"urban" + 0.204*"moderation" + 0.090*"um-11" + 0.032*"um-62"')
(1, '0.250*"maximus" + 0.219*"moderation" + 0.219*"urban" + 0.073*"um-43" + 0.064*"accessory"')
(2, '0.273*"convenience" + 0.262*"urban" + 0.251*"maximus" + 0.026*"uc-50" + 0.026*"uc-55"')
(3, '0.262*"mix" + 0.253*"maximus" + 0.180*"productivity" + 0.090*"uc-21" + 0.082*"season"')
(4, '0.341*"uc-41" + 0.241*"maximus" + 0.189*"uc-74" + 0.153*"youth" + 0.072*"uc-32"')

### Sentiment analysis

                             text_combine  sentiment_score sentiment_category
0       Maximus UM-12 Accessory Accessory              0.0            neutral
1       Maximus UM-12 Accessory Accessory              0.0            neutral
2       Maximus UM-12 Accessory Accessory              0.0            neutral
3       Maximus UM-12 Accessory Accessory              0.0            neutral
4       Maximus UM-12 Accessory Accessory              0.0            neutral
...                                   ...              ...                ...
611165        Maximus UE-19 Urban Extreme              0.0            neutral
611166     Maximus UM-01 Urban Moderation              0.0            neutral
611167     Maximus UM-11 Urban Moderation              0.0            neutral
611168    Maximus UC-01 Urban Convenience              0.0            neutral
611169    Maximus UC-69 Urban Convenience              0.0            neutral

[611170 rows x 3 columns]

These techniques help us understand the underlying topics and sentiments in our text data.

## Usage

Install these Needed Python packages including:

Pandas for data manipulation and analysis
NumPy for numerical operations
Matplotlib for basic data visualization
Seaborn for advanced data visualization
KMeans for clustering
StandardScaler for feature scaling
Train-test split for model evaluation
xgboost
ipywidgets
IPython.display
tkinter
sklearn
torch

## Acknowledgment

Thank you to the online Python and Data commuity for their help.
