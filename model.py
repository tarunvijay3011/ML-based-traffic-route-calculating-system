import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#  LOAD & INITIAL CLEANING
df = pd.read_csv("delhi_traffic_features.csv")
df.drop(columns=['Trip_ID', 'start_area', 'end_area'], inplace=True)

#  ENCODING (Using One-Hot for maximum R2 accuracy)
df = pd.get_dummies(df, columns=['day_of_week','weather_condition','road_type','time_of_day','traffic_density_level'], dtype=int)
df.drop_duplicates(inplace=True)
df.fillna(df.median(), inplace=True)

#  OUTLIER REMOVAL (IQR)
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    return data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]

df = remove_outliers(df, "average_speed_kmph")
df = remove_outliers(df, "distance_km")

#  LOG TRANSFORMATION
df["average_speed_kmph_LOG"] = np.log(df["average_speed_kmph"])
df["distance_km_LOG"] = np.log(df["distance_km"])

# DROP original columns to prevent multicollinearity and "fake" R2 inflation
df.drop(columns=['distance_km', 'average_speed_kmph'], inplace=True)

# 5. DATA SPLITTING
X = df.drop(["average_speed_kmph_LOG"], axis=1)
y = df["average_speed_kmph_LOG"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  FEATURE PIPELINE (Polynomials then Scaling)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()
X_train_final = scaler.fit_transform(X_train_poly)
X_test_final = scaler.transform(X_test_poly)

# FINAL MODEL TRAINING (Ridge)
model = Ridge(alpha=1.0)
model.fit(X_train_final, y_train)

#  PERFORMANCE CHECK
train_r2 = model.score(X_train_final, y_train)
test_r2 = model.score(X_test_final, y_test)
print(f"Final Model Results:\nTrain R2: {train_r2:.4f}\nTest R2: {test_r2:.4f}")

# EXPORT WEIGHTS & EQUATION
# Map coefficients back to their polynomial feature names
feature_names = poly.get_feature_names_out(X.columns)
weights_df = pd.DataFrame({
    'Feature': feature_names,
    'Weight (Coefficient)': model.coef_
})

# Save weights to CSV
weights_df.to_csv("delhi_traffic_weights.csv", index=False)
# Save Intercept
with open("model_intercept.txt", "w") as f:
    f.write(str(model.intercept_))

# Save the entire "Prediction Bundle" (for future use)
model_bundle = {
    "model": model,
    "scaler": scaler,
    "poly": poly,
    "original_features": list(X.columns)
}
joblib.dump(model_bundle, "traffic_prediction_model.pkl")

print("\n--- Model Assets Exported ---")
print("1. delhi_traffic_weights.csv (Feature Weights)")
print("2. model_intercept.txt (The 'b' in y=mx+b)")
print("3. traffic_prediction_model.pkl (Complete Model File)")

# LEARNING CURVE VISUALIZATION
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train_final, y_train, cv=5, scoring='r2', 
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score', marker='o')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-Val Score', marker='s')
plt.title('Final Learning Curve Verification')
plt.xlabel('Training Samples')
plt.ylabel('R2 Score')
plt.legend()
plt.grid(True)
plt.show()
