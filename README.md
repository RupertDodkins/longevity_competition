# ICR - Identifying Age-Related Conditions
## Use Machine Learning to detect conditions with measurements of anonymous characteristics


This is a Python project that focuses on identifying age-related conditions using machine learning algorithms. Kaggle competition can be found here https://www.kaggle.com/competitions/icr-identify-age-related-conditions. The project involves data preprocessing, model training, evaluation, and generating a submission file.

## Project Structure
The project consists of a single Python script with the following sections:

Importing Libraries and Data: The required libraries are imported, including pandas, numpy, and scikit-learn. The training and test datasets are loaded using `pd.read_csv()`.

Exploratory Data Analysis: Basic exploratory data analysis is performed by displaying the first few rows of the training and test datasets using `df.head()` and `test_df.head()`.

Data Preprocessing: Non-numeric columns are identified using `df.select_dtypes(include=['object'])`.columns. The non-feature columns "Id" and "EJ" are dropped from the training dataset using `df.drop(columns=['Id', 'EJ'])`. The feature set `X` and the target labels `y` are defined.

Feature Scaling and Imputation: The feature set is imputed using the mean strategy and standardized using `SimpleImputer` and `StandardScaler` from scikit-learn, respectively.

Model Training and Evaluation: Several classification models are trained using the training dataset. The models include Logistic Regression, Support Vector Machine, K-Nearest Neighbors, Decision Tree, Naive Bayes, and Random Forest. The models are trained using `model.fit()` and evaluated using log loss score with `log_loss()`. The results are stored in a DataFrame `results_df_standardized`.

XGBoost Model: An XGBoost classifier is trained using the training dataset, and log loss score is calculated for evaluation.

Ensemble Model: An ensemble model is created using the best models obtained from XGBoost and Random Forest. The ensemble model is trained and evaluated using the same methodology as the individual models.

Generating Submission: The trained ensemble model is used to make probability predictions on the test dataset. The missing values in the test dataset are imputed and standardized similar to the training dataset. The predictions are then saved in a submission file named `submission.csv` using `pd.DataFrame` and `to_csv()`.

## Usage
To run this project, you need to have Python and the required libraries installed. Follow these steps:

Install Python: Download and install Python from the official website (https://www.python.org) based on your operating system.

Install Dependencies: Open a terminal or command prompt and navigate to the project directory. Run the following command to install the required dependencies:

```pip install pandas numpy scikit-learn xgboost```

Download the Dataset: Download the training and test datasets and place them in the project directory or specify the correct file paths in the trainfile and testfile variables.

Run the Script: Execute the Python script using the following commands:


```pip install notebook```

```jupyter nbconvert --execute train.ipynb```

Results and Submission: After running the script, you can examine the results displayed on the console, including the log loss scores of the individual models and the ensemble model. The submission file `submission.csv` will be generated in the project directory, containing the predictions for the test dataset.

Feel free to modify the code as per your requirements and experiment with different models, hyperparameters, and evaluation metrics to enhance the performance.

## License
This project is released under the MIT License. Feel free to use and modify the code for personal or commercial purposes.