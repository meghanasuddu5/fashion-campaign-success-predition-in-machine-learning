# Fashion Campaign Success Prediction using Machine Learning

This project predicts the **success of fashion marketing campaigns** using machine learning. It helps businesses make **data-driven decisions** by estimating campaign performance before launch based on key parameters.


## Problem Statement
Fashion brands often invest in marketing campaigns without knowing whether they will be effective. This can lead to poor engagement and financial loss. A predictive system is needed to assess campaign success in advance using historical data.


## Objective
To build a machine learning model that predicts whether a fashion marketing campaign will be **successful or not**, based on campaign-related inputs.


## Solution Overview
A classification-based machine learning model is trained on campaign data using features such as discount, campaign duration, and marketing channel.  
The trained model is integrated into a **Flask web application** where users can input campaign details and receive instant predictions with confidence scores.


## Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Flask  
- Joblib  
- HTML, CSS  


## Features Used
- Discount Percentage  
- Campaign Duration (in days)  
- Marketing Channel (encoded categorical feature)  


## Model Details
- Model Type: Classification  
- Output:
  - `1` → Campaign likely to be successful  
  - `0` → Campaign may not be successful  
- Probability scores generated using `predict_proba`

## Application Workflow
1. User enters campaign details through the web interface  
2. Input data is validated and preprocessed  
3. Features are encoded  
4. Data is passed to the trained ML model  
5. Prediction and confidence score are displayed  






