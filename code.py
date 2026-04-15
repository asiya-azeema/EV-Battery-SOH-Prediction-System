# understand data
import pandas as pd
import glob

path = r"C:/Users/azoos/Downloads/Real EV dataset(1)/Real EV dataset/*.csv"

files = glob.glob(path)

print("Files found:", files)

dataframes = []

for file in files:
    df = pd.read_csv(file)
    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)

print(data.head())
print("Dataset shape:", data.shape)
print(data.columns)
print(data.isnull().sum())

# Next Step: Apply the Full Data Quality Pipeline
#1️⃣ Check Missing Values
print(data.isnull().sum())

#2️⃣ Visualize Missing Values
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(data.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

#3️⃣ KNN Imputation (Fill Missing Values)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
imputed_data = imputer.fit_transform(data)
import pandas as pd
data_imputed = pd.DataFrame(imputed_data, columns=data.columns)
print(data_imputed.head())

#4️⃣ Outlier Detection (Isolation Forest)
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05)
outliers = iso.fit_predict(data_imputed)
clean_data = data_imputed[outliers == 1]
print("Rows before cleaning:", len(data_imputed))
print("Rows after removing outliers:", len(clean_data))

#5️⃣ Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clean_data)
scaled_df = pd.DataFrame(scaled_data, columns=clean_data.columns)
print(scaled_df.head())

#6️⃣ Save Clean Dataset
scaled_df.to_csv("clean_ev_battery_dataset.csv", index=False)

#TRAINING ML MODEL
#Step 1: Define Features and Target
X = clean_data[['SOH(OCV)', 'mileage']]
y = clean_data['SOH']
#X → input features
#y → target variable (SOH)
#Step 2: Split Dataset (Train/Test)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#80% training
#20% testing
#Step 3: Train ML Model

#We will use Linear Regression (simple and good for this dataset).

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Model trained successfully!")
#Step 4: Make Predictions
y_pred = model.predict(X_test)

Check predicted values:

print(y_pred[:5])
#step 5: Evaluate Model Performance
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

#If R² is close to 1, the model predicts SOH well.

#Example output:
#Mean Squared Error: 0.002
#R2 Score: 0.91

#That means 91% prediction accuracy.

#Step 6: Compare Actual vs Predicted
import pandas as pd

results = pd.DataFrame({
    "Actual SOH": y_test,
    "Predicted SOH": y_pred
})

print(results.head())
#Step 7: Visualization
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual SOH")
plt.ylabel("Predicted SOH")
plt.title("Actual vs Predicted Battery SOH")
plt.show()

#If points lie near the diagonal line, the model is good.


Final Step: Save the Trained Model (Model Deployment Concept)

Use Joblib to save the model.

Step 1: Install joblib (if needed)
pip install joblib
Step 2: Save the Model

After training your model, run:

import joblib

joblib.dump(model, "battery_soh_model.pkl")

print("Model saved successfully!")

This creates a file:

battery_soh_model.pkl

This file contains your trained AI model.

Step 3: Load the Model Later

If someone wants to use the model later:

import joblib

loaded_model = joblib.load("battery_soh_model.pkl")

prediction = loaded_model.predict([[0.85, 50000]])

print("Predicted SOH:", prediction)

Example input:

SOH(OCV) = 0.85
Mileage = 50000

Output:

Predicted SOH = 0.80

#DEPLOYMENT USING STREAMLIT

Step 1: Install Streamlit
pip install streamlit
Step 2: Create a File

Create a new file called:

app.py
Step 3: Deployment Code

code inside app.py.

import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("battery_soh_model.pkl")

st.title("EV Battery SOH Prediction System")

st.write("Enter battery parameters to predict State of Health")

# Inputs
soh_ocv = st.number_input("SOH(OCV)", min_value=0.0, max_value=1.0, value=0.8)
mileage = st.number_input("Mileage", min_value=0, value=40000)

# Prediction button
if st.button("Predict SOH"):
    
    input_data = np.array([[soh_ocv, mileage]])
    
    prediction = model.predict(input_data)
    
    st.success(f"Predicted Battery SOH: {prediction[0]:.3f}")
Step 4: Run the App

Open terminal and run:

streamlit run app.py

It will open a local web app in your browser.

Example:

http://localhost:8501

Now your AI model works like a real EV battery prediction system.
