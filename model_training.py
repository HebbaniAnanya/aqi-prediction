import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
dataset = pd.read_csv(r"C:\Users\MuraliHebbani\OneDrive\Documents\ani\AQI-merge-tag.csv")


dataset.columns = dataset.columns.str.strip()
dataset['Place'] = dataset['Place'].str.strip()


dataset['date'] = pd.to_datetime(dataset['date'], format='%d-%m-%Y', dayfirst=True)


dataset['year'] = dataset['date'].dt.year
dataset['day_of_week'] = dataset['date'].dt.dayofweek
dataset['is_weekend'] = (dataset['date'].dt.dayofweek >= 5).astype(int)
dataset = dataset.drop(columns=['date'])


# Ensure all feature columns are numeric
for col in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']:
    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')


# Handle missing values
imputer = SimpleImputer(strategy='mean')
dataset[['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']] = imputer.fit_transform(dataset[['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']])


# Analyze distribution of target variable
sns.countplot(x='AQI_bucket_calculated', data=dataset)
plt.title('Distribution of AQI Bucket')
plt.show()


# Split into features and target
X = dataset[['Place', 'Station', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'year', 'day_of_week', 'is_weekend']]
y = dataset['AQI_bucket_calculated']


# Standard scaling
scaler = StandardScaler()
X.loc[:, ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']] = scaler.fit_transform(X[['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']])


# Label encoding for categorical features
place_encoder = LabelEncoder()
station_encoder = LabelEncoder()
X.loc[:, 'Place'] = place_encoder.fit_transform(X['Place'])
X.loc[:, 'Station'] = station_encoder.fit_transform(X['Station'])


# Ordinal encoding for target
order = ["Severe", "Very Poor", "Poor", "Moderate", "Satisfactory", "Good"]
ordinal_encoder = OrdinalEncoder(categories=[order])
y = ordinal_encoder.fit_transform(y.values.reshape(-1, 1)).ravel()


# Hyperparameter tuning using RandomizedSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}


forest_clf = RandomForestClassifier()
random_search = RandomizedSearchCV(forest_clf, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', return_train_score=True)
random_search.fit(X, y)


best_estimator = random_search.best_estimator_


# Cross-validation performance
cv_scores = cross_val_score(best_estimator, X, y, cv=5, scoring='accuracy')
print(f'Cross-validation accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}')


# Feature importance
feature_importances = pd.Series(best_estimator.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importance')
plt.show()


# Save the model and encoders
joblib.dump(best_estimator, 'trained_model.pkl')
joblib.dump(place_encoder, 'place_encoder.pkl')
joblib.dump(station_encoder, 'station_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(ordinal_encoder, 'ordinal_encoder.pkl')


print("Model and encoders saved.")