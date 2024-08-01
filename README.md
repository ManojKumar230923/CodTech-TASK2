
# CodTech-TASK2
# Linear Regression on Samsung Galaxy Data
## Table of Contents
- [Overview](#overview)
- [Key Activities](#key-activities)
- [Technologies](#technologies)
- [Observations / Insights](#observations--insights)
## Overview
**Name:** MANOJ KUMAR

**Company:** CODTECH IT SOLUTIONS

**Domain:** Data Scientist

**ID:** CT8DS698

**Mentor:** Muzammil Ahmed

**Duration:** June, 2024 - August, 2024
### Project
**Project:** Linear Regression on Mobile Phone Data

**Objective:** Perform a linear regression analysis to predict the price of Samsung Galaxy devices based
on various features like 'Rating', 'Spec_score', 'Ram', and 'Battery'.
## Technologies
- Python
- pandas
- scikit-learn
- matplotlib
## Key Activities
### 1. Select Variables
Choose 'Price' as your target variable (dependent variable).
Select numeric features (independent variables) that you believe could influence the price,
such as 'Rating', 'Spec_score', and potentially others like 'Ram', 'Battery', etc., after
converting them into numerical values if necessary.
### 2. Prepare the Data
Convert relevant categorical variables (like 'Ram' or 'Battery') into numerical representations
if they aren't already.
Handle missing values or outliers as needed.
### 3. Split the Data
Split your dataset into training and testing sets. This ensures that you can train the model on
one set and evaluate its performance on unseen data.
### 4. Train the Model
Use a linear regression model from a library such as scikit-learn in Python.
Fit the model to the training data, where you'll provide your selected predictors and the
'Price' as the target variable.
### 5. Evaluate and Visualize
Evaluate the model using metrics like mean squared error (MSE) or R-squared to
understand how well it predicts prices based on the selected features.
Visualize the predicted prices against the actual prices to see how closely they align.
## OUTCOMES
![Predicted VS Actual Price](https://github.com/user-attachments/assets/b43d385c-8aea-4825-8592-acb9ce2197ef)
## Observations / Insights
### Model Performance
- **Mean Squared Error (MSE):** The average of the squares of the errors—that is, the
average squared difference between the estimated values and the actual value.
- **R-squared:** Indicates how well the independent variables explain the variability of the
dependent variable. An R-squared value closer to 1 indicates a better fit.
### Visualizations
Visualizations can help understand the relationship between the predicted and actual prices,
and how well the model is performing.
