# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:53:25 2026

@author: deshp
"""

#===================================
# BUSINESS UNDERSTANDING
#===================================

# BUSINESS PROBLEM

"""
Business Problem:

Telecommunication companies face customer churn, where customers
discontinue their services and switch to competitors. High churn
rates lead to revenue loss, increased customer acquisition costs,
and reduced profitability. Therefore, the company needs to identify
customers who are likely to churn and understand the factors
influencing customer attrition.
"""

# BUSINESS OBJECTIVE

"""
Business Objective:

To analyze customer demographic, account, and service usage
information to identify patterns associated with customer churn.
The objective is to build a predictive model that can classify
customers as likely to churn or not churn, enabling the company
to improve customer retention and reduce revenue loss.
"""

# BUSINESS CONSTRAINTS

"""
Business Constraints:

1. Customer behavior may change over time.
2. The dataset may contain class imbalance between churn and non-churn customers.
3. Some important factors affecting churn may not be available in the dataset.
4. Incorrect predictions may lead to unnecessary retention costs or customer loss.
5. The model should provide reliable predictions to support business decisions.
"""

#===================================
# DATA UNDERSTANDING
#===================================

'''
| Feature          | Description                                    | Type        | Relevance                                    |

|------------------|------------------------------------------------|------------|----------------------------------------------|

| customerID       | Unique customer identifier                     | Categorical | Used for identification only                 |

| gender           | Gender of the customer                         | Categorical | Helps analyze customer demographics          |

| SeniorCitizen    | Indicates whether customer is a senior citizen | Numerical   | Useful for churn analysis                    |

| Partner          | Whether customer has a partner                 | Categorical | May influence customer retention             |

| Dependents       | Whether customer has dependents                | Categorical | Helps understand customer profile            |

| tenure           | Number of months with the company              | Numerical   | Important predictor of churn                 |

| PhoneService     | Whether customer has phone service             | Categorical | Service usage information                    |

| MultipleLines    | Whether customer has multiple lines            | Categorical | Service adoption indicator                   |

| InternetService  | Type of internet service                       | Categorical | Important service-related feature            |

| OnlineSecurity   | Whether online security service is subscribed  | Categorical | May influence churn behavior                 |

| OnlineBackup     | Whether online backup service is subscribed    | Categorical | Service usage information                    |

| DeviceProtection | Whether device protection is subscribed        | Categorical | Service usage information                    |

| TechSupport      | Whether tech support is subscribed             | Categorical | Customer support indicator                   |

| StreamingTV      | Whether streaming TV service is subscribed     | Categorical | Entertainment service usage                  |

| StreamingMovies  | Whether streaming movie service is subscribed  | Categorical | Entertainment service usage                  |

| Contract         | Contract type of customer                      | Categorical | Strong predictor of churn                    |

| PaperlessBilling | Whether paperless billing is enabled           | Categorical | Billing preference                           |

| PaymentMethod    | Customer payment method                        | Categorical | May affect customer retention                |

| MonthlyCharges   | Monthly amount charged to customer             | Numerical   | Important revenue-related feature            |

| TotalCharges     | Total amount charged to customer               | Numerical   | Reflects long-term customer value            |

| Churn            | Whether customer left the service              | Categorical | Target Variable                              |

'''


#=================================================
# EXPLORATORY DATA ANALYSIS FOR TELCO DATASET
#=================================================

#==============================
# IMPORT REQUIRED LIBRARIES
#==============================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#=====================
# LOAD DATASET
#=====================

telecom = pd.read_excel("C:/ASSIGNMENTS/K_means_clustering/Telco_customer_churn.xlsx")

# Display first 5 records
print("First 5 Records:")
print(telecom.head())

#==============================
# BASIC INFORMATION
#==============================

# Shape of dataset
print("\nShape of Dataset:")
print(telecom.shape)
# (7043, 30) 

# Data types
print("\nData Types:")
print(telecom.dtypes)

# Dataset Information
print("\nDataset Information:")
telecom.info()

# Missing value check
print("\nMissing Values:")
print(telecom.isnull().sum())

# Duplicate records
print("\nDuplicate Records:")
print(telecom.duplicated().sum())

"""
Inference:

The Telco dataset contains customer demographic,
service usage, billing, contract and revenue information.

Missing values and duplicate records are checked
before performing clustering analysis.
"""

#==========================
# DESCRIPTIVE STATISTICS
#==========================

print("\nSummary Statistics:")
print(telecom.describe())

"""
Inference:

Summary statistics provide minimum, maximum,
mean, quartiles and standard deviation
for numerical variables.
"""

#=======================================
# BUSINESS MOMENTS
#=======================================

# First Moment - Mean
print(telecom.mean(numeric_only=True))

# Second Moment - Variance & Standard Deviation
print(telecom.var(numeric_only=True))

print(telecom.std(numeric_only=True))

# Third Moment - Skewness
print(telecom.skew(numeric_only=True))

# Fourth Moment - Kurtosis
print(telecom.kurt(numeric_only=True))

"""
Inference:

Mean represents central tendency.

Variance and Standard Deviation indicate
the spread of data.

Skewness indicates asymmetry
in the distribution.

Kurtosis helps identify extreme values
and possible outliers.
"""

#================================
# UNIVARIATE ANALYSIS HISTOGRAM
#================================

telecom.hist(
    figsize=(15,10),
    bins=10
)

plt.suptitle("Histogram of Numerical Variables")
plt.show()

"""
Inference:

Histograms show the distribution
of numerical variables.

Some variables may be skewed and
contain extreme values.
"""

#=================================
# OUTLIER ANALYSIS  BOXPLOT
#=================================

numeric_cols = telecom.select_dtypes(include=np.number)
plt.figure(figsize=(14,6))

sns.boxplot(
    data=numeric_cols
)
plt.xticks(rotation=90)
plt.title("Boxplot of Numerical Variables")
plt.show()

"""
Inference:
Boxplots help identify outliers.

Extreme values may influence
cluster formation.
"""

#=========================================
# CORRELATION ANALYSIS
#=========================================

corr = numeric_cols.corr()
print("\nCorrelation Matrix:")
print(corr)
plt.figure(figsize=(12,8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

"""
Inference:
Correlation heatmap shows the
relationship among numerical variables.

Highly correlated variables
share similar information.
"""

#=================================
# PAIRPLOT
#=================================

selected_cols = numeric_cols.columns[:5]

sns.pairplot(
    telecom[selected_cols],
    diag_kind='kde'
)

plt.show()

"""
Inference:
Pairplot helps visualize relationships
between important numerical variables.

It also shows the distribution pattern
of each variable.
"""

#===============================
# FINAL EDA SUMMARY
#===============================

"""
FINAL EDA SUMMARY

1. Dataset structure and data types were examined.

2. Missing values and duplicate records were checked.

3. Descriptive statistics and business moments
   were analyzed.

4. Histograms were used to study variable distributions.

5. Boxplots were used to identify outliers.

6. Correlation analysis was performed to
   understand relationships among variables.

7. Pairplot was used to study variable interactions.

8. Variables exist on different scales and
   require standardization before clustering.

9.The dataset is suitable for customer segmentation using Agglomerative Clustering.
"""

#=================================================
# DATA PREPROCESSING FOR TELCO DATASET
#=================================================

#======================================
# IMPORT REQUIRED LIBRARIES
#======================================
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#======================================
# CREATE COPY OF DATASET
#======================================

telecom1 = telecom.copy()

telecom1.drop(
    columns=['Customer ID'],
    inplace=True
)

print(telecom1.shape)

"""
Inference:

A copy of the dataset was created for preprocessing.

The 'Customer ID' column was removed because it is
an identifier and does not provide useful information
for clustering.

The dataset is now prepared for the next preprocessing
steps.
"""

#======================================
# MISSING VALUE CHECK
#======================================

print("\nMissing Values:\n")
print(telecom1.isnull().sum())

"""
Inference:

The dataset was checked for missing values before
performing data preprocessing.

Most of the variables do not contain any missing values.

The 'Offer' column contains 3,877 missing values and
the 'Internet Type' column contains 1,526 missing values.

These missing values should be handled before
building the  Agglomerative Clustering model to improve the quality
of clustering.
"""

#======================================
# DUPLICATE RECORD CHECK
#======================================

print("\nDuplicate Records:")
print(telecom1.duplicated().sum())

telecom1.drop_duplicates(inplace=True)

print("\nShape after removing duplicates:")
print(telecom1.shape)

"""
Inference:

Duplicate observations were removed
to improve clustering quality.
"""

#======================================
# DATA TYPE CHECK
#======================================

print("\nData Types:\n")
print(telecom1.dtypes)

#======================================
# LABEL ENCODING
#======================================

le = LabelEncoder()

for col in telecom1.columns:
    
    if telecom1[col].dtype == 'object':
        
        telecom1[col] = le.fit_transform(
            telecom1[col].astype(str)
        )

print("\nEncoded Dataset:")
print(telecom1.head())

"""
Inference:
All categorical variables were converted
into numerical values using Label Encoding.

Agglomerative Clustering requires numerical
input features, so categorical variables
were encoded.
"""

#======================================
# DESCRIPTIVE STATISTICS
#======================================

print("\nSummary Statistics:\n")
print(telecom1.describe())

"""
Inference:

Summary statistics help understand
the transformed dataset before scaling.
"""

#======================================
# FEATURE SCALING
#======================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    telecom1
)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=telecom1.columns
)

print("\nScaled Data:")
print(scaled_df.head())

"""
Inference:

Variables were standardized using
StandardScaler.

This prevents variables with large values
from dominating the clustering process.
"""

#======================================
# VERIFY SCALING
#======================================

print("\nMean after Scaling:\n")
print(scaled_df.mean())

print("\nStandard Deviation after Scaling:\n")
print(scaled_df.std())

"""
Inference:

The mean is approximately 0 and
the standard deviation is approximately 1.

This confirms successful standardization.
"""

#=========================================
# FINAL DATA PREPROCESSING SUMMARY
#=========================================

"""

1. Customer ID column was removed.
2. Missing values and duplicate records were checked.
3. Categorical variables were encoded using Label Encoding.
4. All features were standardized using StandardScaler.
5. Scaling was verified successfully.
6. The dataset is clean and ready for Agglomerative Clustering.

"""

#==========================================
# MODEL BUILDING - AGGLOMERATIVE CLUSTERING
#==========================================

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

#===========================================
# STEP 1: DENDROGRAM
#===========================================

linked = linkage(
    scaled_df,
    method='ward'
)

plt.figure(figsize=(12,6))

dendrogram(
    linked,
    truncate_mode='lastp',
    p=20,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True
)

plt.title("Dendrogram")
plt.xlabel("Clusters")
plt.ylabel("Euclidean Distance")
plt.grid(True)

plt.show()

"""
Inference:

The dendrogram helps determine the
optimal number of clusters.

The largest vertical distance before
merging indicates the suitable number
of clusters.
"""

#===========================================
# STEP 2: BUILD AGGLOMERATIVE MODEL
#===========================================

agg_model = AgglomerativeClustering(
    n_clusters=3,
    metric='euclidean',
    linkage='ward'
)

clusters = agg_model.fit_predict(scaled_df)

# Add Cluster Labels
telecom1["cluster"] = clusters

print("\nCluster Counts:")
print(telecom1["cluster"].value_counts())

"""
Inference:

Each customer has been assigned
to one cluster using Agglomerative
Clustering.
"""

#===========================================
# STEP 3: SILHOUETTE SCORE
#===========================================

score = silhouette_score(
    scaled_df,
    clusters
)

print("\nSilhouette Score :", round(score,4))

"""
Inference:

A higher silhouette score indicates
better cluster separation and compactness.
"""

#===========================================
# STEP 4: CLUSTER PROFILING
#===========================================

cluster_profile = telecom1.groupby("cluster").mean(numeric_only=True)

print("\nCluster Profile:")
print(cluster_profile)

"""
Inference:

Cluster profiling summarizes the
characteristics of each customer group.
"""

#===========================================
# STEP 5: VISUALIZE CLUSTERS
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
plt.title("Agglomerative Clustering")

plt.show()

"""
Inference:

The scatter plot shows the customer
segments formed by Agglomerative
Clustering.
"""

#===========================================
# STEP 6: FINAL MODEL SUMMARY
#===========================================

print("\nAgglomerative Clustering Completed Successfully")

"""
FINAL MODEL SUMMARY

1. A dendrogram was used to determine
   the optimal number of clusters.

2. Agglomerative Clustering grouped
   customers based on hierarchical
   similarity.

3. Silhouette Score was calculated
   to evaluate clustering performance.

4. Cluster profiling was performed
   to understand customer characteristics.

5. The clustering model can support
   customer segmentation and business
   decision-making.
"""






