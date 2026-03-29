# Explainable Credit Risk Decision System with Fairness Analysis

A loan default prediction system built on the Home Credit dataset, combining LightGBM for performance, SHAP for instance-level explainability, a three-tier decision system (approve / review / reject), and a fairness audit across demographic groups. Built with regulatory expectations in mind.

---

## The Problem

Credit scoring is one of the most consequential applications of machine learning. A model that denies a loan affects someone's ability to buy a home, start a business, or handle an emergency. That makes two things non-negotiable:

1. Explainability – the model needs to produce reasons for its decisions, not just a score
2. Fairness – the model should not systematically disadvantage groups based on protected characteristics

This project addresses both. It also reflects the fact that the EU AI Act classifies credit scoring as a high-risk AI system requiring explainability and fairness evaluation.

---

## Results

| Metric    | Logistic Regression | LightGBM |
| --------- | ------------------: | -------: |
| AUC-ROC   |              0.7444 |   0.7407 |
| AUC-PR    |              0.2256 |   0.2277 |
| Recall    |              0.4322 |   0.4272 |
| Precision |              0.2202 |   0.2252 |
| F1        |              0.2918 |   0.2949 |

Evaluated on a stratified 20% holdout set. Thresholds were tuned to optimise F1 rather than using the default 0.5.

LightGBM performs similarly to Logistic Regression rather than clearly outperforming it. This is likely because much of the predictive signal in this dataset is relatively linear and already captured well by Logistic Regression. In addition, the heavy class imbalance limits how much extra performance can be gained from a more complex model.

---

## Key Features

**Three-tier decision system**
Applications are classified as approve, review, or reject using probability thresholds derived from score percentiles. The review category reflects how real lending systems handle borderline cases.

**SHAP explainability**
Each prediction is broken down into feature contributions, allowing individual decisions to be interpreted rather than treated as black boxes.

**Fairness analysis**
Demographic parity and equalised odds are evaluated across gender and age groups. The results show measurable differences between groups, highlighting the need for monitoring and potential mitigation before deployment.

---

## Project Structure

```
credit-risk-dashboard/
│
├── app.py
├── train.py
├── utils.py
├── requirements.txt
│
├── data/
│   └── application_train.csv
│
├── models/
├── plots/
├── dashboard_screenshots/
```

---

## Setup & Running Locally

### 1. Clone the repo

```bash
git clone https://github.com/tabishiqbal03/credit-risk-dashboard.git
cd credit-risk-dashboard
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the data

Download `application_train.csv` from Kaggle and place it inside a `data/` folder.

### 5. Train the models

```bash
python train.py
```

### 6. Run the dashboard

```bash
streamlit run app.py
```

---

## Dashboard Overview

* Model performance: ROC, precision-recall, and score distributions
* Explainability: SHAP feature importance and individual predictions
* Decision system: interactive threshold tuning and decision breakdown
* Fairness analysis: approval rates and recall across demographic groups

---

## Technical Approach

### Why Recall matters

The dataset is highly imbalanced, with only about 8% defaults. Accuracy is therefore misleading. The focus is on recall to ensure that high-risk applicants are identified, while thresholds are adjusted to manage false positives.

### Handling class imbalance

* Logistic Regression uses balanced class weights
* LightGBM uses scale_pos_weight

### SHAP

SHAP values are computed on a representative sample of the test set, providing both global and local explanations.

### Fairness

The model shows differences in approval rates and recall across demographic groups. This does not necessarily indicate intentional bias, but it does highlight areas that would need further investigation before real-world deployment.

---

## Technologies

* Python 3.11
* pandas, numpy
* scikit-learn
* LightGBM
* SHAP
* Streamlit
* matplotlib

---

## Regulatory Context

Credit scoring is considered high-risk under the EU AI Act. This project demonstrates awareness of key requirements such as explainability, human oversight, and fairness evaluation.

---

## Limitations & Future Work

* Only a subset of the available dataset is used
* Fairness analysis is limited to gender and age
* Probability calibration could be improved
* Model performance could be enhanced with additional features and tuning

---

## Author

Tabish Iqbal
GitHub: https://github.com/tabishiqbal03
