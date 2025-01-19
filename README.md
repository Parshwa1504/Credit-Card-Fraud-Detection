 #README: Credit Card Default Prediction

## Overview

This project aims to predict credit card default behavior using machine learning algorithms. The dataset is provided by American Express, and various predictive models including **XGBoost**, **CatBoost**, and **LightGBM** are used to predict whether a customer will default on their credit card payments.

The project includes multiple steps such as:
1. **Data Preprocessing:** Data cleaning, handling missing values, and feature selection.
2. **Exploratory Data Analysis (EDA):** Visualizing and analyzing data patterns to understand the relationship between features and the target variable.
3. **Model Building:** Implementing and training machine learning models like XGBoost, CatBoost, and LightGBM.
4. **Model Evaluation:** Using various metrics such as F1-score, AUC-ROC, and confusion matrix to evaluate model performance.
5. **Prediction and Submission:** Making predictions on the test set and saving them for submission.

## Project Structure

- **`CS584_Project.ipynb`**: The main Python notebook containing the entire workflow for data analysis, model training, evaluation, and submission.
- **`requirements.txt`**: A text file listing the dependencies needed to run the code.
- **`data/`**: A folder that contains the dataset files (train_data.ftr, test_data.ftr, etc.). This folder is not included in the repository but can be added to store your datasets.

## Requirements

To run this project, ensure you have the following dependencies installed:

- **pandas**: For data manipulation.
- **numpy**: For numerical operations.
- **matplotlib** & **seaborn**: For data visualization.
- **plotly**: For interactive plots.
- **lightgbm**: For the LightGBM model.
- **xgboost**: For the XGBoost model.
- **catboost**: For the CatBoost model.
- **sklearn**: For machine learning tools and metrics.

You can install the necessary libraries using `pip`:

```
pip install -r requirements.txt
```

# Instructions
   
1. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   - Download the dataset from the American Express Credit Card Default Prediction competition or use the dataset provided in the project.
   - Place the dataset files in the `data/` folder.

4. Open and run the `CS584_Project.ipynb` notebook in Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook CS584_Project.ipynb
   ```

## Usage

### Data Preprocessing

- The dataset is preprocessed by grouping data based on `customer_ID` to ensure only the latest customer record is considered.
- Missing values are handled using imputation strategies to ensure the data is clean for model training.
  
### Exploratory Data Analysis (EDA)

- Visualizations such as **Correlation Heatmaps** and **KDE Plots** are used to understand relationships between features and the target variable (default vs. non-default).
  
### Model Building

-**XGBoost**, **CatBoost**, and **LightGBM** models are trained on the preprocessed dataset. Each model is optimized using hyperparameters.
- Model evaluation is done using metrics like **F1-Score**, **AUC-ROC**, and **Log Loss**.

### Model Evaluation

- For each model, the performance is assessed based on confusion matrices, ROC curves, and various classification metrics.
- **AUC-ROC** is used to evaluate the models' discriminative power.

### Making Predictions

- After evaluating the models, predictions are made on the test dataset and saved in the submission format.

## Output

1. **Visualizations**: Various plots (e.g., ROC Curves, Confusion Matrices, Feature Importance).
2. **Performance Metrics**: Detailed evaluation of each model (XGBoost, CatBoost, LightGBM).
3. **Submission File**: The final output containing the model predictions for the test dataset (`submission.csv`).

## Future Work

- **Model Improvement**: Further optimization of hyperparameters using grid search or random search.
- **Handling Imbalanced Data**: Experimenting with different resampling techniques to handle class imbalance.
- **Model Interpretability**: Implementing SHAP or LIME to explain model predictions and feature importance.
- **Deployment**: Deploying the model in a web application or API for real-time credit card default predictions.

## Contributing

Feel free to fork this repository, submit issues, and contribute improvements. For any questions or feedback, open an issue or reach out via email.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
