# 📅 Day 3 - Customer Segmentation

## 🎯 Objective

Identify different customer groups based on purchasing behavior using unsupervised machine learning techniques.

---

## ✅ Tasks Completed

### 1. Data Preparation

Loaded the RFM dataset.

Selected features:

- Recency
- Frequency
- Monetary

Standardized data using StandardScaler.

---

### 2. K-Means Clustering

Applied the Elbow Method to determine the optimal number of clusters.

Optimal clusters selected:

```
K = 4
```

Trained the K-Means clustering model.

---

### 3. Model Evaluation

Silhouette Score:

```
0.616
```

This indicates good cluster separation.

---

### 4. Customer Segments

Identified four customer groups:

- 👑 VIP Customers
- 💎 Loyal Customers
- 🛒 Regular Customers
- ⚠️ At Risk Customers

---

### Cluster Summary

 Cluster                 Description .
 VIP Customers         High frequency, highest revenue, recent purchases .
 Loyal Customers       Frequent buyers with high spending .
 Regular Customers     Average purchase behavior .
 At Risk Customers    Low spending and inactive customers .

---

### 5. DBSCAN Clustering

Applied DBSCAN for comparison.

Results:

- 1 Dense Cluster
- 41 Outlier Customers

DBSCAN was useful for identifying anomalies but K-Means provided more meaningful business segments.

---

## 📊 Comparison

 Algorithm       Result 

 K-Means   4 Business-Friendly Clusters 
 DBSCAN    1 Cluster + 41 Outliers 

---

## 📂 Output File

```
data.
│
└── customer_segments.csv
```

---

## 🛠 Technologies Used

- Python
- Scikit-Learn
- StandardScaler
- K-Means
- DBSCAN
- Matplotlib
- Seaborn

---

# 📈 Business Outcome

The customer segmentation model helps businesses:

- Identify VIP customers
- Improve customer retention
- Design personalized marketing campaigns
- Detect inactive customers
- Optimize inventory planning using customer demand patterns

---

# 🚀 Next Step

Day 4:

- Time Series Preparation
- Stationarity Testing
- Prophet Forecasting
- Demand Prediction