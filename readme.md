 # Olist E-commerce Customer Value Optimization & Segmentation

# Project Overview
This project focuses on transforming raw transactional data from Olist, Brazil's largest department store, into actionable business intelligence. The core objective is to understand customer behavior, segment the customer base into meaningful groups, and predict customer monetary value. This enables Olist's marketing team to design highly targeted strategies, improve customer retention, and maximize customer lifetime value (CLTV).

##   Problem Statement
Problem Statement: As an analytics professional for Olist, a Brazilian e-commerce platform, we aim to optimize marketing spend and enhance customer lifetime value. How can we segment Olist's diverse customer base based on their purchasing habits and satisfaction, predict their potential future value, and develop highly tailored marketing strategies to maximize engagement and sales?

The **Goal** is to:

Identify distinct **customer segments** based on purchasing behavior, preferences, and value.

**Characterize each segment** to understand their unique needs and motivations.

Recommend **tailored marketing and product strategies** for each identified segment to improve engagement and sales.



# Business Value & Impact
Understanding and optimizing customer value is paramount for e-commerce growth. This project delivers substantial business value by:

**Optimized Marketing Spend:** Enables targeted campaigns for specific customer segments, leading to more efficient resource allocation and higher ROI.

**Enhanced Customer Lifetime Value (CLTV):** By predicting future customer spending and identifying high-potential customers, Olist can implement proactive retention strategies and nurture valuable relationships.

**Improved Customer Experience:** Insights from customer segmentation and sentiment analysis allow for personalized engagement, addressing specific customer needs and boosting satisfaction.

**Data-Driven Decision Making:**Provides empirical evidence for strategic decisions related to product offerings, promotions, and customer service.


# Data Sources
The project utilized 8+ publicly available datasets from Olist, covering various aspects of their e-commerce operations:

[View Olist Dataset on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

olist_customers_dataset.csv

olist_orders_dataset.csv

olist_order_items_dataset.csv

olist_products_dataset.csv

olist_sellers_dataset.csv

olist_order_reviews_dataset.csv

olist_order_payments_dataset.csv

olist_geolocation_dataset.csv


# Project Goals
This project was structured into four key phases:

**Data Integration & Foundational Cleaning:** Combine disparate datasets to form a unified, clean, and comprehensive analytical base.

**Feature Engineering & Exploratory Data Analysis (EDA):** Create rich, predictive features from raw data and uncover key behavioral patterns, trends, and relationships.

**Customer Segmentation & NLP for Review Insights:** Discover distinct customer groups using unsupervised learning and extract valuable qualitative insights from customer reviews.

**Customer Monetary Value Prediction:** Develop a robust machine learning model to accurately forecast the future monetary contribution of customers.


# Methodology & Key Steps
## **notebook 1: Data Integration & Foundational Cleaning**
Integrated 8+ disparate e-commerce CSVs into a single, comprehensive dataset using Pandas, forming a unified view of customer interactions.

Performed foundational data cleaning, including handling missing values, standardizing data types, and identifying/removing duplicates, to ensure data integrity and reliability for subsequent analysis.

## **notebook 2: Feature Engineering & Exploratory Data Analysis (EDA)**
Engineered over **30 high-impact features** from raw transactional data, including:

**RFM (Recency, Frequency, Monetary)** metrics to quantify customer value.

**Product-centric features:** average price per customer, total items purchased, number of unique product categories/sellers.

**Payment-related features:** average/max installments, payment type indicators.

**Time-based features:** customer tenure.

Geographic features and state-level indicators.

**Conducted in-depth EDA using Matplotlib and Seaborn** to uncover initial patterns, correlations, and anomalies, revealing key behavioral differences and informing feature selection.

## **notebook 3: Customer Segmentation & NLP for Review Insights**
**Leveraged Natural Language Processing (NLP) with NLTK (VADER)** to extract sentiment (positive, negative, compound scores) from 1M+ customer reviews, enriching customer profiles with qualitative feedback.

**Segmented over 100,000 unique customers into 2 distinct behavioral groups using K-Means clustering (validated with a 0.65 Silhouette Score)**.

Characterized each customer segment based on their unique RFM scores, product preferences, payment behaviors, and aggregated review sentiment, providing actionable profiles for targeted marketing.

## **notebook 4: Customer Monetary Value Prediction**
Developed and rigorously optimized a Customer Monetary Value prediction model using advanced regression techniques.

Utilized XGBoost Regressor (tuned via GridSearchCV and K-Fold Cross-Validation with Scikit-learn) for superior predictive performance, alongside RandomForestRegressor and Logistic Regression for baseline comparisons.

Addressed data skewness in monetary values using log1p transformation for improved model learning.

# Key Insights & Results Summary
This project delivered a comprehensive understanding of Olist's customer base and predictive capabilities:

**Customer Segmentation:** Identified distinct groups (e.g., "High-Value Loyalists," "Emerging/At-Risk Customers") enabling personalized marketing strategies for each segment.

**Sentiment Analysis:** Quantified customer satisfaction directly from review text, providing a nuanced view beyond numerical ratings.

**Predictive Accuracy:** The XGBoost Regressor model achieved outstanding performance in predicting customer monetary value:

**R-squared: 0.929** (explaining 92.9% of the variance in customer spending)

**RMSE (Root Mean Squared Error): 73.32 BRL** (average prediction error on the original scale)

**Churn Indication:** Negative sentiment, coupled with certain RFM patterns, correlated with lower future monetary value, indirectly indicating churn risk or disengagement.

# Actionable Business Recommendations
Based on the analytical findings, the following recommendations are proposed to Olist's marketing and product teams:

**Tailored Retention Campaigns:** Develop specific campaigns for each customer segment, focusing on re-engagement for "Emerging/At-Risk Customers" and loyalty programs for "High-Value Loyalists."

**Leverage Sentiment Data:** Use sentiment scores from reviews to proactively address customer dissatisfaction and identify product/service areas for improvement.

**Targeted Promotions:** Promote annual plans at stations predominantly used by casual riders (identified via EDA) and offer shorter membership trials near tourist hotspots to convert casuals into members.

**Personalized Product Recommendations:** Utilize segmented customer preferences and historical purchases to offer highly personalized product recommendations, enhancing customer experience and driving sales.

# Tools & Technologies Used
Programming Language: Python

**Data Manipulation:** pandas, numpy
**Data Visualization:** matplotlib.pyplot, seaborn
**Machine Learning:** scikit-learn (for preprocessing, model building, **evaluation, clustering), XGBoost
**Natural Language Processing (NLP):** nltk (VADER sentiment analysis)
**Notebook Environment:** Jupyter Notebook
**Model Persistence:** joblib (for saving models/scalers)
**Version Control:** Git & GitHub

