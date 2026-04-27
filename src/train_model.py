import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import math

print("Loading saved features...")
features = np.load('data/features/features.npy')
labels   = np.load('data/features/labels.npy')
yields   = np.load('data/features/yields.npy')
print(f"Loaded {len(features)} images!")

def generate_weather(disease_name):
    weather_profiles = {
        "Late_blight":  {"temp": (10,22), "humidity": (80,95), "rainfall": (5,20)},
        "Early_blight": {"temp": (24,32), "humidity": (60,80), "rainfall": (2,10)},
        "healthy":      {"temp": (20,28), "humidity": (40,60), "rainfall": (0, 3)},
    }
    profile = weather_profiles["healthy"]
    for key in weather_profiles:
        if key.lower() in disease_name.lower():
            profile = weather_profiles[key]
            break
    return np.array([
        np.random.uniform(*profile["temp"]),
        np.random.uniform(*profile["humidity"]),
        np.random.uniform(*profile["rainfall"]),
        np.random.uniform(0, 30),
    ])

print("Generating weather features...")
weather_feats = np.array([generate_weather(l) for l in labels])

print("Scaling weather features...")
scaler = StandardScaler()
weather_scaled = scaler.fit_transform(weather_feats)

print("Fusing CNN + weather features...")
X = np.concatenate([features, weather_scaled], axis=1)
y = yields
print(f"Final feature shape: {X.shape}")

print("Splitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"Training: {len(X_train)} images | Testing: {len(X_test)} images")

print("Training XGBoost model... (this may take a few minutes)")
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)
print("Training complete!")

print("Evaluating model...")
preds = model.predict(X_test)
rmse  = math.sqrt(mean_squared_error(y_test, preds))
r2    = r2_score(y_test, preds)
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.3f}")

print("Saving model and scaler...")
pickle.dump(model,  open('models/xgb_model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl',    'wb'))
print("All done! Model saved to models/xgb_model.pkl")