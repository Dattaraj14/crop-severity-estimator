# Development guide

## Run the app

From the repository root, install the current dependencies and start Streamlit:

    python -m pip install -r requirements.txt
    streamlit run app/streamlit_app.py

The app UI can be opened without the training dataset because it uses the tracked XGBoost model and scaler. The first prediction may require internet access to download torchvision's pretrained ResNet-50 weights. Runtime packages are currently mixed with training and plotting packages in one unpinned requirements file; splitting and pinning them is a useful follow-up.

## Recreate the current training artifacts

1. Download PlantVillage from the [Kaggle dataset page](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).
2. Place the class folders under **data/raw/plantvillage dataset/color/**.
3. Create **data/features/**.
4. From the repository root, run:

       python src/extract_features.py
       python src/train_model.py
       python src/explain.py

Feature extraction runs ResNet-50 on every image and may take a long time on CPU. The scripts use relative paths and expect to be launched from the repository root. The extraction and explanation scripts perform work at module execution time, so do not import them as library modules.

These commands reproduce the current prototype pipeline; they do not validate real yield-loss prediction. See [Data and model notes](data-and-model.md).

## Current project boundaries

- **app/streamlit_app.py** contains both the UI and inference code.
- **src/** contains standalone feature extraction, training, synthetic label, and SHAP scripts.
- **models/** contains the saved regressor, scaler, and plot.
- **data/** is local-only and ignored by Git.

A future cleanup could move reusable inference code into an installable package, put runnable scripts under **scripts/**, and keep the Streamlit file focused on UI. That refactor should preserve the current model's feature order and preprocessing unless the model is retrained.

## Contribution guidance

When changing inference, document preprocessing and feature order because the saved XGBoost artifact expects exactly 2,048 ResNet values followed by four scaled weather values. When changing training, document data sources and split logic. Do not present a metric change as a real-world improvement unless the target is measured and the evaluation supports that claim.

There is currently no automated test suite or license file in the repository. Add a license only after the repository owner chooses one.
