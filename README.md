Project Overview
This project is part of the final course assignment in Machine Learning Applications using Python. It focuses on implementing and evaluating classification and regression models, including data preprocessing, handling missing values, feature engineering, and model evaluation.

Requirements
The project was completed using Python in a Jupyter Notebook and includes the following major components:

Data Preprocessing
Classification and Regression Model Training
Hyperparameter Tuning
Model Evaluation and Selection
Key Skills Demonstrated
1. Data Import and Preprocessing
Imported and loaded the dataset using pandas.
Checked and handled duplicate values using DataFrame.duplicated() and drop_duplicates().
Dealt with missing values using various imputation techniques.
Converted categorical variables into dummy variables (one-hot encoding).
Normalized and standardized the data to improve model performance.
2. Exploratory Data Analysis (EDA)
Analyzed distributions, checked for outliers, and performed summary statistics.
Visualized data distributions and category frequencies to gain insights.
Checked class balance and applied class balancing techniques.
3. Classification Models
Used XGBoost, Random Forest, and LGBM for classification.
Implemented a custom neural network using Keras for additional classification analysis.
Performed hyperparameter tuning using GridSearchCV to optimize model performance.
Evaluated classification models using metrics such as Accuracy, Precision, Recall, and F1-score.
4. Regression Models
Built regression models using Linear Regression, Polynomial Regression, Lasso, and Ridge.
Evaluated regression models with metrics including R², RMSE, MAE, and MAPE.
Used polynomial transformations and regularization techniques to improve model fit and prevent overfitting.
5. Advanced Model Evaluation and Selection
Compared multiple models and selected the best model based on performance on both training and test sets.
Explained the concepts of overfitting and underfitting and applied cross-validation to reduce overfitting.
Project Structure
MTA_Final_term_exercise_2024.ipynb: Jupyter Notebook containing the code and explanations for each task.
data/: Folder containing datasets used in the project.
images/: Folder containing visualizations and plots generated during EDA and model analysis.
Getting Started
Clone this repository:


Libraries and Tools Used
Python (version 3.x)
Pandas for data manipulation and analysis.
NumPy for numerical operations.
Matplotlib and Seaborn for data visualization.
Scikit-learn for machine learning models, hyperparameter tuning, and evaluation metrics.
Keras for building and training neural networks.
Conclusion
This project showcases my proficiency in handling real-world machine learning tasks, including:

End-to-end pipeline creation (data preprocessing, model training, and evaluation).
Effective handling of classification and regression tasks.
Optimization techniques using hyperparameter tuning.
For further exploration, additional models and techniques like Ensemble Methods or Deep Learning Architectures can be applied to improve the results.

