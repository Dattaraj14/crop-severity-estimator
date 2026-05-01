\# Crop Disease Severity Estimator



🌐 \*\*Live Demo:\*\* https://crop-severity-estimator-gbqokyxjsyrqyesauyknsb.streamlit.app/



A machine learning system that predicts \*\*yield loss percentage\*\* from plant leaf images combined with weather data. Unlike typical disease classifiers, this project predicts the actual financial impact of crop disease on farmers.



\## What Makes This Unique

\- Fusion architecture — combines CNN image features with weather data

\- Regression output — predicts continuous yield loss %, not just a disease label

\- SHAP explainability — shows WHY the model made each prediction

\- Live web app — interactive demo built with Streamlit



\## Results

| Metric | Score |

|--------|-------|

| R2 Score | 0.906 |

| RMSE | 7.40 |



\## Tech Stack

\- \*\*PyTorch + ResNet-50\*\* — CNN feature extraction (2048 features per image)

\- \*\*XGBoost\*\* — yield loss regression model

\- \*\*SHAP\*\* — explainable AI visualization

\- \*\*Streamlit\*\* — web app interface

\- \*\*PlantVillage Dataset\*\* — 54,305 leaf images across 38 disease classes



\## How It Works

1\. ResNet-50 reads a leaf image and produces 2048 numbers describing it

2\. Weather data (temperature, humidity, rainfall, wind) is added — 4 more numbers

3\. XGBoost takes all 2052 numbers and predicts yield loss percentage

4\. SHAP explains which features drove the prediction



\## How To Run Locally



\*\*Install dependencies:\*\*

\*\*Run the web app:\*\*

\## SHAP Analysis

Humidity was found to be among the top 6 most influential features out of 2052 total — beating thousands of CNN image features. This confirms weather genuinely drives yield loss predictions.



!\[SHAP Plot](models/shap\_plot.png)



\## Project Structure

\## Dataset

PlantVillage Dataset — 54,305 leaf images across 38 disease classes

Source: kaggle.com/datasets/abdallahalidev/plantvillage-dataset

