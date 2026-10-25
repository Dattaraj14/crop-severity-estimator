# Data and model notes

## What the current model receives

The app combines a leaf image and four user-entered weather values:

- Temperature in °C
- Humidity in %
- Rainfall in mm
- Wind in km/h

The image is converted to RGB, resized to 224 × 224, normalized with ImageNet statistics, then passed through the feature-extractor portion of ResNet-50. This produces 2,048 image features. The four weather values are scaled with the saved StandardScaler and appended to form a 2,052-feature vector for XGBoost.

The ResNet-50 weights are requested from torchvision's ImageNet weights on first model use. The repository does not contain a copy of those weights.

## Dataset and generated values

The feature extraction script expects PlantVillage class folders at **data/raw/plantvillage dataset/color/**. The dataset is not included in this repository. The README links to [the Kaggle dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset); check the dataset's terms before using it.

The current training target is synthetic:

- **src/yield_labels.py** maps each disease class to a hand-authored base percentage and adds small deterministic noise.
- **src/train_model.py** generates weather values from broad disease-name profiles and random ranges. These values are not observed weather for the images.
- Training and evaluation split individual images at random. Images from the same disease class can occur in both train and test sets.

Consequently, the model learns associations in this constructed dataset. The README's reported R² and RMSE do not measure prediction accuracy on measured yields or prove usefulness for farm decisions. Treat output as a prototype score only. Avoid calling R² “accuracy.”

## UI behavior to know

The app's disease selector is currently not passed into the model. The selected value is shown in the result text, but prediction uses the uploaded image features and scaled weather values. The model is not a separate 38-class disease classifier.

The app does not compute SHAP explanations per uploaded image. **models/shap_plot.png** is a static summary produced by **src/explain.py**.

## Artifacts and reproducibility

Tracked model-related files:

- **models/xgb_model.pkl** — trained XGBoost regressor.
- **models/scaler.pkl** — scaler for the four weather inputs.
- **models/shap_plot.png** — offline summary plot.

The feature arrays and raw images are excluded from Git by **.gitignore**. A fresh clone therefore cannot reproduce training without manually downloading and arranging PlantVillage data. The extraction script also expects **data/features/** to exist before it saves arrays.

The target noise uses Python's built-in hash value to seed a random generator; hash values can differ between Python processes unless hash randomization is controlled. The training script's generated weather values are also not seeded. Repeated training runs may produce different targets, weather features, and metrics.

## Evaluation needed for a real yield-loss claim

A future real-world claim would need a sourced, documented dataset with measured yield loss aligned to each image and weather observation. The evaluation should split by source, plant, field, or another meaningful grouping so near-duplicate or same-class samples do not leak across the split. Report the target definition, units, sample counts, split method, and a baseline alongside metrics. Until then, keep the app and README explicit that the target is synthetic.
