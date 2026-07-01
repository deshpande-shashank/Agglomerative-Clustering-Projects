# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 23:00:28 2026

@author: deshp
"""

#=========================================
# BUSINESS UNDERSTANDING
#=========================================

'''
1. Business objective 
To identify groups of states having similer crime patterns using Agglomerative 
Clustring.

the objectives of project is to analyze crime patterns across differnt 
states and segment states with similer crime characterstics into distinct
groups. The identified clusters can help law enforcement agencies and 
policymakers develop effective crime prevention and resource allocation
strategies.  
'''

'''

 Constraints

1. The dataset has only a few crime-related features.

2. Some states may have very high or very low crime values, which can affect 
the results.

3. The results depend on the quality of the data.

4. Choosing a different number of clusters may give different results.

5. The dataset shows crime rates only for a particular period.

6. The clusters show similar states, but they do not tell the exact reason for 
the crime rates.

'''

#=========================================
# 2. DATA UNDERSTANDING
#=========================================

'''

| Feature  | Description                    | Type        | Relevance                                   |
| -------- | ------------------------------ | ----------- | ------------------------------------------- |

| State    | Name of the state              | Categorical | Used to identify states                     |

| Murder   | Murder rate in the state       | Numerical   | Important for crime analysis                |

| Assault  | Assault rate in the state      | Numerical   | Important for crime analysis                |

| UrbanPop | Percentage of urban population | Numerical   | Helps understand population characteristics |

| Rape     | Rape rate in the state         | Numerical   | Important for crime analysis                |

'''

#==================================
# EXPLORATARY DATA ANALYSIS 
#=================================

#=============================
# import required libraries
#=============================

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from feature_engine.outliers import Winsorizer
#=========================================
# LOAD DATASET
#=========================================

crime = pd.read_csv("C:/ASSIGNMENTS/K_means_clustering/crime_data.csv")

# Display First 5 Records
print("First 5 Records:",crime.head())

#=========================================
# BASIC INFORMATION
#=========================================

# Shape of The Dataset 
print("\nShape of The Dataset:",crime.shape)
# Shape of The Dataset: (50, 5)

# Data Types
print("\nCheck Datatype:",crime.dtypes)
# dtype: object

# Dataset Information
print("\nDataset Information:",crime.info())

# Missing Value Check
print("\nCheck Missing Value:",crime.isnull().sum())
# No missing Values in this Dataset

# Duplicate Records
print("\nDuplicate Record:",crime.duplicated().sum())

'''
Inference:
Dataset contains Crime-Related information of states.
No Missing Values and Duplicate Records should be present.
'''

#=========================================
# BUSINESS MNOMENT DECISION
#==========================================

print("\nSummery Statics:",crime.describe())

'''
Inference:
Summery statics provide minimum,maximum,
mean, and quartile values of each variable.
large differences between min and max indicate variation.
'''

#=========================================
# BUSINESS MOMENT DECISION
#=========================================

# First Moment - Mean
print(crime.mean(numeric_only = True))

# Second Moment - Variance & Standard Deviation
print(crime.std(numeric_only = True))

print(crime.std(numeric_only = True))

# Third Moment - Skewness
print(crime.skew(numeric_only = True))

# Fourth Moment - Kurtosis  
print(crime.kurt(numeric_only = True))

'''
Inference:
The analysis shows that crime levels differ among statics and some extreme
values may be present. Since the variables are on different Scales, data
standardization is necessary before performing agglomerative clustring. 
'''

#=========================================
# UNIVARATE ANALYSIS (HISTOGRAM)
#=========================================

crime.hist(figsize = (10, 8), bins = 10)
plt.suptitle("Histogram of Crime Variables")
plt.show()

'''
Inference:
Histogram shows the distribution of each variable.
crime variables have different range and distributions.
some variables slightly skewed.

'''    

#=========================================
# OUTLIER ANALYSIS (BOXPLOT)
#=========================================

numeric_cols = crime.select_dtypes(include=np.number)

plt.figure(figsize=(10,6))
sns.boxplot(data = numeric_cols)
plt.title("Boxplot of Crime Variables")
plt.show()

'''
Inference:
Boxplots helps identify extreme values (outliers).
some outlier may contain outliers which can affect clustring. 
'''

#=========================================
# BIVARIATE ANALYSIS (CORELATION MATRIX)
#=========================================

corr = numeric_cols.corr()
print("\nCorrelation Matrix:\n",corr)

plt.figure(figsize=(8,6))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
    )
plt.title("Correlation Heatmap")
plt.show()

'''
Inference:
The correlation matrix shows the relationship among the crime variables.
Some variables have positive correlations, while others have weaker
relationships. This helps understand the association between variables
before clustering.
'''

#=========================================
# OUTLIER TREATMENT - WINSORIZATION
#=========================================
numeric_cols = crime.select_dtypes(include=np.number)

print(numeric_cols.columns)

# Select numerical columns
numeric_cols = ['Murder', 'Assault', 'UrbanPop', 'Rape']

# Winsorizer
winsor = Winsorizer(
    capping_method= 'iqr',
    tail='both',
    fold=1.5,
    variables=numeric_cols
    )

crime[numeric_cols] = winsor.fit_transform(crime[numeric_cols])

print("Winsorization Applied Succesfully")

'''
Inference:
Winsorization reduced the effect of extreme values (outliers) without removing
any records from the dataset. This makes the data more balanced and suitable
for Agglomerative Clustering.

'''

#=========================================
# PAIRPLOT  
#=========================================

sns.pairplot(
    crime[['Murder','Assault','UrbanPop','Rape']],
    diag_kind='kde',
    )
plt.show()

'''
Inference:
The pairplot helps visualize the distribution and relationships among
the variables. It provides a better understanding of the data before
performing clustering analysis.
'''

#=========================================
# FINAL EDA CONCLUSION
#=========================================

'''
Final EDA Summery:

1. No Missing Values or duplicate records were found .
2. Crime rate is very significantly among states.
3. Some variables contain outliers.
4. Positive relationship exist among crime variables.
5. Variables are measured on different scales.
6. Data standardization is required before applying agglomerative clustring. 
7. The dataset is suitable for clustring states based on crime patterns.

'''

#=============================================================
# DATA PREPROCESSING FOR CRIME DATA (AGGLOMERATIVE CLUSTRING)
#=============================================================

#=====================================
# IMPORT REQUIRED LIBRARIES
#=====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

#======================================
# LOAD DATASET
#======================================

crime = pd.read_csv("C:/ASSIGNMENTS/K_means_clustering/crime_data.csv")

print("Initial Shape:",crime.shape)
# Initial Shape: (50, 5)

print(crime.head)

#========================================
# DATA TYPE CHECK
#========================================

print(crime.info())
'''
Inference:
'state' is a categorical variable.
All Remaining variables will be used for clustring.
only numerical varibles will be used for clustring.

'''

#=========================================
# MISSING VALUE CHECK
#=========================================

print("\nMissing Value Check:\n",crime.isnull().sum())

'''
Inference:
No missing values were found in the dataset.
The data is complete and ready for further preprocessing.
If, founf They should be handled before clustring.
'''

#=========================================
# DUPLICATE RECORD CHECK
#=========================================
print("Duplicate Record Check:\n",crime.duplicated().sum())
# No duplicate is present in this dataset

# Duplicate Record is present in the datast 
# Remove this duplicate record 
crime.drop_duplicates(inplace = True)
print("Shape after duplicate Removal:",crime.shape)
# Shape after duplicate Removal: (50, 5)

'''
Inference:
No duplicate records were found in the dataset.
Therefore, no records were removed before clustering.
'''

#=========================================
# REMOVE IDENTIFICATION COLUMN
#=========================================

crime1 = crime.iloc[:,1:]

print("\nShape after Removing identifer column:",crime1.shape)
# Shape after Removing identifer column: (50, 4)
print(crime1.head)

'''
Inference:
'Unnamed:0' contains state names and is used only
for identification purpose.

since agglomerative clustring works only with Numerical 
variables, the identifier column is removed.

'''

#=========================================
# DESCRIPTIVE STATISTIC
#=========================================

print("\nSummery Statics:\n",crime1.describe())

'''
Inference:
The varibales have different range and scales.
Standardization is required before clustring.
'''

#=========================================
# FEATURE SCALING 
#=========================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(crime1)
scaled_df = pd.DataFrame(
    scaled_data,
    columns = crime1.columns
)

print("\nScaled Data",scaled_df.head())

'''
Inference:
StandarsScaler transforms the data so that
all variables have a common scale.
The prevents large-valued variables from 
dominating the clustring process.
'''

#=========================================
# VARIFY SCALING 
#=========================================
print("\nMean After Scaling:\n",scaled_df.mean())

print("\nStandard Deviation After Scaling:\n",scaled_df.std())

'''
Inference:
The scaled data has a mean close to 0 and a standard deviation close to 1.
This confirms that feature scaling was applied successfully and the data is
ready for Agglomerative Clustering.
'''

#=========================================
# FINAL DATA PREPROCESSING SUMMERY
#=========================================

'''
FINAL DATA PREPROCESSING SUMMERY

1. Dataset loaded successfully.
2. Data types varified.
3. Missing values checked.
4. Duplicate records removed.
5. Identification column removed.
6. Numerical variables standardized.

The dataset is now ready for agglomerative clustring.

'''

#=========================================
# MODEL BUILDING AGGLOMERATIVE CLUSTRING
#=========================================

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

#=========================================
# DENDROGRAM
#=========================================

linked = linkage(
    scaled_df,
    method='ward'
)

plt.figure(figsize=(12,6))

dendrogram(
    linked,
    truncate_mode='lastp',
    p=20,
    leaf_rotation = 90,
    leaf_font_size = 10,
    show_contracted=True
)

plt.title("Dendrogram")
plt.xlabel("Clusters")
plt.ylabel("Euclidean Distance")
plt.grid(True)
plt.show()

'''
Inference:
The dendrogram helps determine the 
optimal number of clusters.

The largest vertical distance before
merging indicates the suitable number 
of clusters.

'''

#=========================================
# BUILD AGGLOMERATIVE MODEL 
#=========================================

agg_model = AgglomerativeClustering(
    n_clusters= 3,
    metric='euclidean',
    linkage='ward'
)

clusters = agg_model.fit_predict(scaled_df)

# Add Cluster Labels
crime1["cluster"] = clusters

print("\nCluster counts:")
print(crime1["cluster"].value_counts())

'''
Inference:

The model successfully divided the states into 3 clusters based on similer 
crime characterstics. The cluster labels and counts show how the states a 
distributed across the identified groups.  
'''

#=========================================
# SILHOUETTE SCORE
#=========================================

score = silhouette_score(
    scaled_df,
    clusters
)
print("\nSilhouette score :",round(score,4))
# Silhouette score : 0.3123

'''
Inference:
The silhouette score indicates the quality of the clusters.
The obtained score suggests a reasonable clustering structure for the dataset.
'''

#=========================================
# CLUSTER PROFILING
#=========================================\

cluster_profile = crime1.groupby("cluster").mean(numeric_only = True)

print("\nCluster Profile:")
print(cluster_profile)

'''
Inference:
The cluster profile summerize the avarage values of each cluster. It helps
identify and compare the characterstics of different groups of states.
'''

#===========================================
# VISUALIZE CLUSTERS
#===========================================

plt.figure(figsize=(8,6))

plt.scatter(
    scaled_df.iloc[:,0],
    scaled_df.iloc[:,1],
    c=clusters,
    cmap='viridis',
    s=40
)

plt.xlabel(scaled_df.columns[0])
plt.ylabel(scaled_df.columns[1])
plt.title("Agglomoretive Clustring")
plt.show()

'''
Inference:
The scatter plot shows the clusters formed by the Agglomerative Clustering
model. States with similar characteristics are grouped together into the same
cluster.
'''

#=========================================
# FINAL MODEL SUMMARY
#=========================================

print("\nAgglomerative Clustering Completed Successfully")


'''
Inference:

The Agglomerative Clustering model successfully grouped the states
into 3 clusters based on similar crime characteristics. The results
help identify different crime patterns among the states and support
further analysis.

'''

#=========================================
# BUSINESS BENEFITS
#=========================================

"""
BUSINESS BENEFITS

1. Helps identify states with similar crime patterns.

2. Supports better planning and allocation of police resources.

3. Assists policymakers in developing effective crime prevention strategies.

4. Enables authorities to focus on high-risk areas for improved public safety.

5. Supports data-driven decision-making for crime analysis and resource management.

"""


