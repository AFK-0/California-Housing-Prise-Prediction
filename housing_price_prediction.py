import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading California Housing dataset...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame
df['MedHouseVal'] = housing.target

print("Dataset loaded successfully. Shape:", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())P

print("\n--- Data Exploration and Cleaning ---")

print("\nMissing values per column:")
print(df.isnull().sum())
print("No explicit missing values found or complex outlier handling required for this dataset.")
print("Data is considered clean for initial modeling.")

print("\nDescriptive statistics:")
print(df.describe())

plt.figure(figsize=(10, 6))

plt.subplot(2, 3, 1)
sns.histplot(df['MedInc'], kde=True)
plt.title('Distribution of Median Income')

plt.subplot(2, 3, 2)
sns.histplot(df['HouseAge'], kde=True)
plt.title('Distribution of House Age')

plt.subplot(2, 3, 3)
sns.histplot(df['AveRooms'], kde=True)
plt.title('Distribution of Average Rooms')

plt.subplot(2, 3, 4)
sns.histplot(df['Population'], kde=True)
plt.title('Distribution of Population')

plt.subplot(2, 3, 5)
sns.histplot(df['MedHouseVal'], kde=True, color='red')
plt.title('Distribution of Median House Value (Target)')

plt.tight_layout()

print("\nCorrelation Matrix:")
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features and Target')

features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
target = 'MedHouseVal'

X = df[features]
y = df[target]

print(f"\nFeatures selected for the model: {features}")
print(f"Target variable: {target}")

print("\n--- Model Training ---")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape (X_train): {X_train.shape}")
print(f"Testing data shape (X_test): {X_test.shape}")

model = LinearRegression()

print("Training the Linear Regression model...")
model.fit(X_train, y_train)
print("Model training complete.")

print("\n--- Model Evaluation ---")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual House Values')
plt.ylabel('Predicted House Values')
plt.title('Actual vs. Predicted House Values')
plt.grid(True)
plt.show()