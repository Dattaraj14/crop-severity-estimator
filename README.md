# Crop Disease Severity Estimator

🌐 **Live demo:** [Streamlit app](https://crop-severity-estimator-gbqokyxjsyrqyesauyknsb.streamlit.app/)

This project is an experimental image-and-weather demo. It extracts visual features from a leaf image with an ImageNet-pretrained ResNet-50 and combines them with four weather inputs. An XGBoost regressor then returns a percentage based on the project's current **synthetic yield-loss proxy**.

> **Research limitation:** the current target is generated from disease-name rules, not measured crop yields. The reported scores describe this generated target on a random image split. They are not evidence that the app predicts real-world yield loss. See [Data and model limitations](docs/data-and-model.md).

## Try the app

The deployed app is the easiest way to explore the interface. On the first prediction after a fresh start, the app may need to download ResNet-50's pretrained weights. The app asks for a leaf image and weather values. Its disease selector is currently display-only; it does not alter the model prediction.

To run locally, use Python and install the packages listed in **requirements.txt**:

    python -m pip install -r requirements.txt
    streamlit run app/streamlit_app.py

The first prediction may require internet access so torchvision can fetch the pretrained ResNet-50 weights. To understand the app's resource and cold-start behavior, see [Deployment](docs/deployment.md).

## How it works

1. Resize and normalize the uploaded image to 224 × 224 pixels.
2. Use ResNet-50 to extract 2,048 image features.
3. Scale the four weather values using the saved StandardScaler.
4. Concatenate the image and weather features into a 2,052-value input.
5. Use the saved XGBoost model to estimate the synthetic percentage target.

The app caches the model resources within a running Streamlit process. Its SHAP plot is a saved, offline summary image; the app does not calculate a new SHAP explanation for each prediction.

## Reported training results

The existing project README reports **R² = 0.906** and **RMSE = 7.40**. The training script prints metrics on a random 80/20 image split. Because labels and weather are generated from disease classes, these scores should be treated as prototype results on synthetic data—not as field accuracy or verified agricultural performance.

## Repository map

    app/streamlit_app.py   Streamlit user interface and inference flow
    src/                   Feature extraction, training, labels, and SHAP script
    models/                Saved XGBoost model, weather scaler, and SHAP plot
    data/                  Local dataset and generated features; not tracked by Git
    docs/                  Data/model, development, and deployment notes

## Dataset and training

The feature extraction script expects the PlantVillage dataset under:

    data/raw/plantvillage dataset/color/

Download the dataset from [Kaggle's PlantVillage dataset page](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) and arrange its class folders at that path. Create **data/features/** before running the script; the script writes feature arrays there.

From the repository root, the current scripts run in this order:

    python src/extract_features.py
    python src/train_model.py
    python src/explain.py

Feature extraction processes the full image collection and can take a long time on CPU. Training and explanation require the generated arrays. The data and feature files are intentionally excluded from Git. Details and limitations are in [Data and model limitations](docs/data-and-model.md).

## Contributing

Start with [Development guide](docs/development.md) to understand the current project boundaries and commands. Before describing a model change as an improvement, document its data source and evaluate it on a split that avoids images from the same class or source appearing on both sides.

## License

No license file is currently included. Ask the repository owner before reusing or redistributing the code or model artifacts.
