import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import pickle

print("=" * 60)
print("MEDICAL COST PREDICTION USING MACHINE LEARNING")
print("=" * 60)

# Load dataset
print("\n[1] Loading dataset...")
data = pd.read_csv('insurance.csv')
print(f"Dataset shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())

# Data Preprocessing - Label Encoding
print("\n[2] Data Preprocessing (Label Encoding)...")
lab = LabelEncoder()
data['sex'] = lab.fit_transform(data['sex'])
data['smoker'] = lab.fit_transform(data['smoker'])
data['region'] = lab.fit_transform(data['region'])
print("After encoding:")
print(data.head())

# Data Exploration - Graphs
print("\n[3] Generating EDA graphs...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='smoker', data=data)
plt.title('Smoker Distribution')
plt.subplot(1, 2, 2)
sns.countplot(x='sex', data=data)
plt.title('Gender Distribution')
plt.tight_layout()
plt.savefig('eda_graphs.png')
print("Graph saved as 'eda_graphs.png'")
plt.show()

# Split Data
print("\n[4] Splitting data (70% train, 30% test)...")
x = data.iloc[:, data.columns != 'charges']
y = data.iloc[:, data.columns == 'charges']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=0)
print(f"Training set: {xtrain.shape[0]} rows")
print(f"Testing set: {xtest.shape[0]} rows")

# Model 1: Random Forest Regressor
print("\n[5] Training Models...")
print("-" * 40)
print("1. Random Forest Regressor")
rf = RandomForestRegressor(n_estimators=300, random_state=0)
rf.fit(xtrain, ytrain)
y_pred_rf = rf.predict(xtest)
r2_rf = r2_score(ytest, y_pred_rf)
print(f"   R² Score: {r2_rf:.6f}")

# Model 2: Linear Regression
print("\n2. Linear Regression")
lr = LinearRegression()
lr.fit(xtrain, ytrain)
y_pred_lr = lr.predict(xtest)
r2_lr = r2_score(ytest, y_pred_lr)
print(f"   R² Score: {r2_lr:.6f}")

# Model 3: Decision Tree Regressor
print("\n3. Decision Tree Regressor")
dt = DecisionTreeRegressor(random_state=0)
dt.fit(xtrain, ytrain)
y_pred_dt = dt.predict(xtest)
r2_dt = r2_score(ytest, y_pred_dt)
print(f"   R² Score: {r2_dt:.6f}")

# Model 4: SVR
print("\n4. Support Vector Regressor (SVR)")
svr = SVR()
svr.fit(xtrain, ytrain)
y_pred_svr = svr.predict(xtest)
r2_svr = r2_score(ytest, y_pred_svr)
print(f"   R² Score: {r2_svr:.6f}")

# Summary
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"{'Model':<25} {'R² Score':<15}")
print("-" * 40)
print(f"{'Random Forest':<25} {r2_rf:.6f}")
print(f"{'Linear Regression':<25} {r2_lr:.6f}")
print(f"{'Decision Tree':<25} {r2_dt:.6f}")
print(f"{'SVR':<25} {r2_svr:.6f}")

# Save the best model (Random Forest)
pickle.dump(rf, open('model.pkl', 'wb'))
print("\n✓ Best model (Random Forest) saved as 'model.pkl'")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)