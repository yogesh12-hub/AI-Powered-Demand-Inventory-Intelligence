# 📅 Day 2 - Data Cleaning & Feature Engineering

## 🎯 Objective

Prepare the raw retail dataset for machine learning by cleaning inconsistent data, engineering meaningful features, and creating customer-level metrics.

---

## ✅ Tasks Completed

### 1. Data Cleaning

The following cleaning steps were performed:

- Removed cancelled orders from the forecasting dataset.
- Stored cancelled transactions separately for business analysis.
- Removed rows with missing CustomerID.
- Removed duplicate records.
- Removed records with negative Quantity.
- Removed records with zero or negative UnitPrice.

---

### 2. Feature Engineering

Created the following features:

- Revenue = Quantity × UnitPrice
- Year
- Month
- Day
- Weekday
- Hour
- IsWeekend

These features improve forecasting and customer behavior analysis.

---

### 3. RFM Analysis

Generated customer-level RFM metrics:

- Recency
- Frequency
- Monetary

Created:

- R Score
- F Score
- M Score
- RFM Score
- Customer Segment

---

### 4. Rolling Statistics

Prepared time-series features:

- Daily Revenue
- 7-Day Rolling Mean
- 30-Day Rolling Mean
- Rolling Standard Deviation

These features will be used for demand forecasting.

---

### 5. Data Validation

Validated the cleaned dataset by checking:

- Missing values
- Positive Quantity
- Positive UnitPrice
- Revenue values
- CustomerID completeness

---

## 📂 Output Files

```
data/
│
├── cleaned_retail.csv
├── cancelled_orders.csv
├── rfm_dataset.csv
└── daily_sales.csv
```

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

