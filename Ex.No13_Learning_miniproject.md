# Ex.No: 10 Learning – Use Supervised Learning  
### DATE: 25/10/2025                                                                         
### REGISTER NUMBER : 212223040120

### AIM: 
To train a supervised learning model using Random Forest Regressor to predict medical insurance charges based on patient details.

###  Algorithm:
1. Import necessary libraries such as pandas, sklearn, numpy and matplotlib.

2. Load the dataset and explore the structure using head() and info().

3. Visualize the data using suitable graphs to understand relationships.

4. Convert categorical data into numerical form using one-hot encoding.

5. Split the dataset into training and testing sets.

6. Create the Random Forest Regressor model and train it with training data.

7. Predict medical charges for test data and evaluate performance using RMSE and R² score.

8. Plot Actual vs Predicted graph and residual plot.

9. Display feature importance values.

### Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# NEW LIBRARIES FOR GRAPHING
import matplotlib.pyplot as plt
import seaborn as sns
```
```
# The file path is just the name of the file you uploaded
file_path = '/content/insurance.csv'
df = pd.read_csv(file_path)

# Print the first 5 rows
print("--- Data Head ---")
print(df.head())

# Print a summary to check data types
print("\n--- Data Info ---")
df.info()
```
### Output:
<img width="1159" height="645" alt="image" src="https://github.com/user-attachments/assets/9468a76e-f821-456d-a177-168633275d92" />

```
# Set the style for our plots
sns.set_style('whitegrid')

plt.figure(figsize=(30, 13))
sns.histplot(df['charges'], kde=True)
plt.title('Distribution of Medical Charges', fontsize=16)
plt.xlabel('Charges ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()
```
### Output:
<img width="1767" height="682" alt="image" src="https://github.com/user-attachments/assets/3b519274-fa38-42c0-a8be-01ca11be77de" />
```
plt.figure(figsize=(30, 8))
sns.scatterplot(x='age', y='charges', data=df, hue='smoker', palette='bright', s=100)
plt.title('Age vs. Charges (Color-coded by Smoker Status)', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Charges ($)', fontsize=12)
plt.legend(fontsize=12)
plt.show()
```
### Output:
<img width="1776" height="664" alt="image" src="https://github.com/user-attachments/assets/201b9732-da3b-4a71-99fe-e765be79a9e9" />
```
plt.figure(figsize=(25, 8))
sns.scatterplot(x='bmi', y='charges', data=df, hue='smoker', palette='coolwarm', s=100)
plt.title('BMI vs. Charges (Color-coded by Smoker Status)', fontsize=16)
plt.xlabel('BMI', fontsize=12)
plt.ylabel('Charges ($)', fontsize=12)
plt.legend(fontsize=12)
plt.show()
```
### Output:
<img width="1776" height="676" alt="image" src="https://github.com/user-attachments/assets/f1c48f95-d034-4dad-8d1d-22ae5a12668b" />
```
# Convert categorical columns into dummy/indicator variables
df_processed = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# See our new, AI-ready data
print("--- Processed Data Head ---")
print(df_processed.head())
```
### Output:
<img width="1234" height="403" alt="image" src="https://github.com/user-attachments/assets/a44993c9-c134-410b-8283-aa8e0bcf1544" />
```
# Our target (y) is what we want to predict
y = df_processed['charges']

# Our features (X) are all the *other* columns
X = df_processed.drop('charges', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Teach (fit) the model on the training data
model.fit(X_train, y_train)

print("Random Forest model training complete!")

# Make predictions on the test data
y_pred = model.predict(X_test)

# --- Graph 4: Actual vs. Predicted Costs ---

# Create a scatter plot of Actual vs. Predicted values
plt.figure(figsize=(25, 9))
plt.scatter(y_test, y_pred, alpha=0.5) # alpha makes points slightly transparent

# Add a line for perfect predictions (y=x)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)

plt.title('Actual vs. Predicted Medical Charges (Random Forest)', fontsize=16)
plt.xlabel('Actual Charges ($)', fontsize=12)
plt.ylabel('Predicted Charges ($)', fontsize=12)
plt.grid(True)
plt.show()
```
### Output:
<img width="1754" height="662" alt="image" src="https://github.com/user-attachments/assets/afe72150-1775-4e6b-8424-3bb7558e98f2" />
```
# --- Graph 5: Residuals Plot ---

# Calculate residuals (errors)
residuals = y_test - y_pred

# Create a scatter plot of Predicted Values vs. Residuals
plt.figure(figsize=(25, 8))
plt.scatter(y_pred, residuals, alpha=0.5)

# Add a horizontal line at y=0 (zero error)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)

plt.title('Residuals vs. Predicted Values (Random Forest)', fontsize=16)
plt.xlabel('Predicted Charges ($)', fontsize=12)
plt.ylabel('Residuals ($)', fontsize=12)
plt.grid(True)
plt.show()
```
### Output:
<img width="1752" height="646" alt="image" src="https://github.com/user-attachments/assets/7d1f4a01-d6e9-4178-8a1b-4831adbe6f6d" />

```
# Make predictions on the test data
y_pred = model.predict(X_test)

# --- 1. Calculate Error (RMSE) ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# --- 2. Calculate R-squared (R2) ---
r2 = r2_score(y_test, y_pred)

print(f"--- Your Enhanced AI Project Results ---")
print(f"Algorithm Used: Random Forest Regressor")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print("--------------------------------------------------------")
print(f"R-squared (R2 Score): {r2 * 100:.4f}%")
print("--------------------------------------------------------")

# --- 3. Show Feature Importance ---
print("\n--- Feature Importance ---")
features = X.columns
importances = model.feature_importances_

importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)
```
### Output:
<img width="1215" height="580" alt="image" src="https://github.com/user-attachments/assets/e519ad39-6d77-4e9f-b013-a498fe710d3b" />

### Result:
Thus the system was trained successfully and the prediction was carried out.
