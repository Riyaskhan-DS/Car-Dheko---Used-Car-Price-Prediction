# Car Dheko - Used Car Price Prediction

**Developed by**: **Mohammed Riyaskhan S**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Skills Takeaway](#skills-takeaway)
- [Domain](#domain)
- [Problem Statement](#problem-statement)
- [Project Scope](#project-scope)
- [Project Approach](#project-approach)
- [Results](#results)
- [Project Evaluation Metrics](#project-evaluation-metrics)
- [Technical Tags](#technical-tags)
- [Dataset](#dataset)
- [Project Deliverables](#project-deliverables)
- [Project Guidelines](#project-guidelines)

---

## Project Overview

This project focuses on developing a **machine learning model** to predict the prices of used cars accurately. Leveraging historical data and various car features such as make, model, year, fuel type, and transmission type, the model aims to enhance the customer experience on Car Dheko by providing instant price predictions. The final solution includes an interactive web application developed with **Streamlit** that allows users to input car details and view price predictions in real-time.

---

## Skills Takeaway

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Machine Learning Model Development
- Price Prediction Techniques
- Model Evaluation and Optimization
- Model Deployment
- Streamlit Application Development
- Documentation and Reporting

---

## Domain
**Automotive Industry, Data Science, Machine Learning**

---

## Problem Statement

### Objective:
Imagine you are a data scientist at Car Dheko. Your task is to improve customer experience and streamline the pricing process by building an accurate and user-friendly Streamlit tool that predicts used car prices based on various features. The goal is to deploy this as an interactive web application for seamless use by both customers and sales representatives.

---

## Project Scope

We have access to a historical dataset of used car prices from CarDekho, covering various cities and attributes. As a data scientist, the task is to clean, preprocess, and transform this data, followed by developing a machine learning model that accurately predicts used car prices based on input features. This model is then deployed as a web app with Streamlit for interactive use.

---

## Project Approach

### Data Processing
1. **Import and Concatenate**:
   - Load data from various cities in unstructured formats and convert them into a structured format.
   - Add a "City" column to each dataset to identify the source, then concatenate all datasets into a single dataframe.

2. **Handling Missing Values**:
   - Fill or drop missing values in both numerical and categorical columns using appropriate imputation methods.

3. **Standardizing Data Formats**:
   - Clean data entries to remove units (e.g., "70 kms" to "70") and ensure consistency in data types.

4. **Encoding Categorical Variables**:
   - Apply one-hot encoding for nominal variables and label/ordinal encoding where applicable.

5. **Normalizing Numerical Features**:
   - Scale features using techniques like Min-Max Scaling or Standard Scaling.

6. **Removing Outliers**:
   - Use IQR or Z-score analysis to detect and handle outliers, improving model accuracy.

### Exploratory Data Analysis (EDA)
1. **Descriptive Statistics**: Calculate summary statistics (mean, median, mode, etc.) to understand data distribution.
2. **Data Visualization**: Generate scatter plots, histograms, and heatmaps to identify patterns and correlations.
3. **Feature Selection**: Identify significant features affecting car prices using correlation analysis and domain knowledge.

### Model Development
1. **Train-Test Split**: Split data into training and testing sets with a standard ratio (e.g., 80-20).
2. **Model Selection**: Test various algorithms including Linear Regression, Decision Trees, Random Forests, and Gradient Boosting.
3. **Model Training**: Train selected models with cross-validation for robust performance.
4. **Hyperparameter Tuning**: Optimize model parameters using Grid Search or Random Search.

### Model Evaluation
1. **Performance Metrics**: Evaluate models using MAE, MSE, and R-squared.
2. **Model Comparison**: Compare models and select the best-performing one.

### Optimization
1. **Feature Engineering**: Enhance model by creating/adjusting features based on domain knowledge.
2. **Regularization**: Apply L1 (Lasso) or L2 (Ridge) regularization to avoid overfitting.

### Deployment
1. **Streamlit Application**: Deploy the final model with Streamlit, allowing real-time input and predictions.
2. **User Interface**: Design a user-friendly interface with clear instructions.

---

## Results

- Developed a high-accuracy machine learning model for predicting used car prices.
- Created comprehensive analysis and visualizations.
- Deployed an interactive Streamlit application for instant price prediction.

---

## Project Evaluation Metrics

- **Model Performance**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R-squared
- **Data Quality**:
  - Completeness and accuracy of preprocessed data.
- **Application Usability**:
  - User feedback and satisfaction with Streamlit app.
- **Documentation**:
  - Clarity and thoroughness of project report and code comments.

---

## Technical Tags

**Data Preprocessing**, **Machine Learning**, **Price Prediction**, **Regression**, **Python**, **Pandas**, **Scikit-Learn**, **EDA**, **Streamlit**, **Model Deployment**

## RESULTS;

![Screenshot 2024-10-28 194021](https://github.com/user-attachments/assets/b6348876-4fb2-4266-829f-20193882186b)
![Screenshot 2024-10-28 193938](https://github.com/user-attachments/assets/73d835ba-9307-4c62-9c02-38114d894d10)
![Screenshot 2024-10-28 194113](https://github.com/user-attachments/assets/f2f3d858-96ed-41af-871d-f45eeaedfd2d)
![Screenshot 2024-10-28 194049](https://github.com/user-attachments/assets/dff44822-2daf-4de9-aa8b-8a7f27b6b448)



