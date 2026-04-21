# E-commerce-clickstream-prediction
Predicting e-commerce purchase intent using interpretable machine learning (Logistic Regression &amp; LightGBM with SHAP)
## 📌 Project Overview

In the fast-growing world of e-commerce, understanding **customer behavior** is critical for increasing conversions and revenue. This project focuses on analyzing **clickstream data** (user browsing activity) to predict whether a user will make a purchase during a session.

Using **machine learning and explainable AI**, this project builds a predictive system that not only forecasts purchase intent but also explains *why* users are likely to buy.

---

## ❗ Problem Statement

E-commerce platforms face a major challenge:

* Millions of users browse products but **do not complete purchases**
* Traditional analytics tools fail to capture **complex user behavior**
* Many machine learning models act as **black boxes**, making them difficult to trust and use in business decisions

As highlighted in the study, businesses struggle with:

* Identifying **high-intent customers in real time**
* Understanding **user behavior patterns**
* Making **data-driven decisions** due to lack of model transparency 

---

## 💡 Proposed Solution

This project solves the problem by building an **interpretable machine learning pipeline** that:

✔ Predicts whether a user session will lead to a purchase
✔ Uses **early session behavior (prefix-based prediction)** for real-time insights
✔ Provides **explainability using SHAP values**
✔ Compares:

* Logistic Regression (interpretable baseline)
* LightGBM (high-performance model)

The system ensures:

* No data leakage (real-world simulation)
* Scalable and reproducible pipeline
* Business-friendly insights

---

## ⚙️ Methodology

The project follows a structured data science pipeline:

1. **Data Collection**

   * Clickstream session data (user interactions)

2. **Data Preprocessing**

   * Handling missing values
   * Sessionization (grouping user activity)
   * Feature scaling & encoding

3. **Feature Engineering**

   * Behavioral features (clicks, views, duration)
   * Price-related statistics
   * Product diversity & interaction patterns

4. **Modeling**

   * Logistic Regression (baseline)
   * LightGBM (advanced model)

5. **Evaluation Metrics**

   * ROC-AUC
   * Precision-Recall AUC (important for imbalance)

6. **Explainability**

   * SHAP (global + local explanations)
   * Feature importance analysis

---

## 📊 Key Findings / Results

After implementing the models, the project revealed:

* ✅ Interpretable models can achieve **high predictive performance**
* ✅ Early-session behavior is a strong indicator of purchase intent
* ✅ Key drivers of purchase include:

  * Product exploration depth
  * Price variation
  * Session engagement
* ✅ Explainability helps uncover **why users convert**

According to the analysis:

* Behavioral patterns like **product variety and browsing depth significantly influence purchase decisions** 

---

## 🚀 Business Impact

This project provides real-world value for e-commerce platforms:

* 🎯 Identify high-intent users early
* 💰 Increase conversion rates through targeted strategies
* 🧠 Enable data-driven decision-making
* 🔍 Improve transparency and trust in AI systems
* 📈 Support personalization, pricing, and marketing strategies

---

## 🧰 Tech Stack

* **Python** (Pandas, NumPy, Scikit-learn)
* **LightGBM**
* **SHAP (Explainable AI)**
* **Matplotlib / Seaborn**
* **Jupyter Notebook**
* **Power BI / Tableau**



## 🔮 Future Improvements

* Integrate real-time streaming data
* Add deep learning models (RNN / Transformers)
* Include user demographics for better personalization
* Deploy as a web-based decision support system

---

## 📌 Conclusion

This project demonstrates that:

> It is possible to build **accurate, interpretable, and practical machine learning models** for predicting online purchase behavior.

By combining predictive power with explainability, this approach bridges the gap between **data science and real-world business decision-making**.

---

## 👨‍💻 Author

# Surya Rajan Saravanan

---
