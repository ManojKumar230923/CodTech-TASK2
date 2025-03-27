#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# In[78]:


# Load the dataset
df = pd.read_csv(r'C:\Users\WINDOWS\Downloads\archive\mobile phone price prediction.csv')
print(df.head())


# In[79]:


selected_features = ['Rating','Spec_score','Ram','Battery']
target_variable ='Price'


# In[80]:


# Clean 'Ram', 'Battery' and 'Price' columns
df['Ram'] = df['Ram'].str.extract('(\d+.\d+|\d+)').astype(float)
df['Battery'] = df['Battery'].str.extract('(\d+.\d+|\d+)').astype(float)
df['Price'] = df['Price'].str.replace(',', '').astype(float)
df.head()


# In[81]:


# Split the data into training and testing sets
X = df[selected_features]
y = df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)


# In[82]:


# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[83]:


# Make predictions on the test set
y_pred = model.predict(X_test)
# Optionally, you can create a DataFrame to compare actual vs.predicted prices
predictions_df = pd.DataFrame({'Actual_Price': y_test,
'Predicted_Price': y_pred})
print(predictions_df.head()) # Display the first few predictions forcomparison


# In[84]:


# Calculate Mean Squared Error and R-Squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


# In[85]:


# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Price')
plt.show()


# In[86]:


# Example new data (features of a new smartphone)
new_phone_features = pd.DataFrame({
'Rating': [4.5],
'Spec_score': [70],
'Ram': [6],
'Battery': [5000]
})
# Predict price for the new data using the trained model
predicted_price = model.predict(new_phone_features)
print(f'Predicted Price for the New Smartphone:${predicted_price[0]:,.2f}')


# In[ ]:




