import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Loading model and features...")
model    = pickle.load(open('models/xgb_model.pkl', 'rb'))
scaler   = pickle.load(open('models/scaler.pkl',    'rb'))
features = np.load('data/features/features.npy')
labels   = np.load('data/features/labels.npy')

from src.yield_labels import get_yield_loss

print("Generating weather features for 100 samples...")
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

sample_features = features[:100]
sample_labels   = labels[:100]

weather_feats  = np.array([generate_weather(l) for l in sample_labels])
weather_scaled = scaler.transform(weather_feats)
X_sample = np.concatenate([sample_features, weather_scaled], axis=1)

feature_names = [f'cnn_{i}' for i in range(2048)] + \
                ['temperature', 'humidity', 'rainfall', 'wind']

print("Running SHAP analysis... (may take 1-2 minutes)")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

print("Generating SHAP summary plot...")
shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=feature_names,
    max_display=15,
    show=False
)
plt.tight_layout()
plt.savefig('models/shap_plot.png', dpi=150, bbox_inches='tight')
print("Done! SHAP plot saved to models/shap_plot.png")