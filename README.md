🚗 EV Battery SOH Prediction System
📌 Overview

This project presents an Automated Data Quality & Machine Learning Pipeline to predict the State of Health (SOH) of Electric Vehicle (EV) batteries.

The system processes raw EV battery data, performs data cleaning and preprocessing, and uses machine learning models to accurately estimate battery health. A user-friendly web interface is built using Streamlit for real-time predictions.

🎯 Key Features

✔ Automated data quality pipeline
✔ Missing value handling using KNN Imputation
✔ Outlier detection using Isolation Forest
✔ Feature scaling for improved model performance
✔ Machine Learning models:

Linear Regression
Random Forest

✔ Model evaluation using:

Mean Squared Error (MSE)
R² Score

✔ Interactive web app for prediction

📊 Dataset Information
Source: EV battery dataset (commercial electric buses)
Total records: 662 rows
Features:
SOH – State of Health (Target Variable)
SOH(OCV) – SOH estimated using Open Circuit Voltage
Mileage – Distance traveled by vehicle
⚙️ Tech Stack
Python
Pandas & NumPy
Scikit-learn
Matplotlib & Seaborn
Streamlit
🔄 Project Workflow
Raw EV Dataset
      ↓
Data Integration (Merge CSV files)
      ↓
Data Quality Check
      ↓
KNN Imputation
      ↓
Outlier Detection (Isolation Forest)
      ↓
Feature Scaling
      ↓
Machine Learning Model
      ↓
Battery SOH Prediction
      ↓
Model Deployment (Streamlit)
📈 Model Performance
Metric	Value
Mean Squared Error	0.0017
R² Score	0.53

The model shows moderate performance due to limited features but demonstrates the effectiveness of data preprocessing.

🖥️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/EV-Battery-SOH-Prediction.git
cd EV-Battery-SOH-Prediction
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the App
streamlit run app/app.py


🚀 Applications
EV battery health monitoring
Predictive maintenance
Fleet management systems
Smart energy management

🔮 Future Scope
Include more features (temperature, voltage, current)
Apply deep learning models
Integrate real-time IoT data
Deploy as mobile/web application

👩‍💻 Author
Asiyamath Azeema
Artificial Intelligence & Data Science Engineer
4BP22AD008
BEARYS INSTITUTE OF TECHNOLOGY

⭐ Acknowledgment
This project demonstrates the real-world application of data science in the Electric Vehicle domain, focusing on improving battery health prediction and maintenance strategies.
