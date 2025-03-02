# California Housing Prices Prediction

## Introduction
This project aims to predict the median house values in California districts based on various features such as median income, housing age, location, and more. The goal is to build a machine learning model that can accurately estimate housing prices, which can be valuable for real estate stakeholders.

Key steps include:
- Data exploration and preprocessing
- Exploratory Data Analysis (EDA)
- Model selection and training
- Hyperparameter tuning
- Performance evaluation

## Dataset
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices) and contains the following features:
- **Numerical Features**: longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value
- **Categorical Feature**: ocean_proximity

**Target Variable**: `median_house_value`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/california-housing-prices-analysis.git
   cd california-housing-prices-analysis
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices) and place `housing.csv` in the project directory.

## Usage
Run the Jupyter Notebook `California_Housing_Prices_Analysis.ipynb` to execute the entire analysis:

```bash
jupyter notebook California_Housing_Prices_Analysis.ipynb
```

### Key Steps:
1. **Data Preprocessing**:
   - Handle missing values in `total_bedrooms` using median imputation.
   - Clip outliers using IQR method.
   - Feature engineering (e.g., `rooms_per_household`).
   - Encode categorical variables (One-Hot Encoding for `ocean_proximity`).
   - Standardize numerical features.

2. **Exploratory Data Analysis (EDA)**:
   - Analyze correlations between features and the target variable.
   - Visualize geographical distribution of house values.
   - Explore feature distributions and relationships.

3. **Model Training**:
   - Train three models: Linear Regression, Decision Tree, and Random Forest.
   - Evaluate using metrics: R², MSE, RMSE, MAE.

4. **Hyperparameter Tuning**:
   - Use `GridSearchCV` to optimize model parameters.
   - Compare performance before and after tuning.

## Results
### Model Performance Comparison (Test Set)
| Model                | R² Score | RMSE      | MAE       |
|----------------------|----------|-----------|-----------|
| Linear Regression    | 0.599    | 71,231    | 50,644    |
| Decision Tree        | 0.699    | 61,646    | 39,944    |
| **Random Forest**    | **0.811**| **48,899**| **31,913**|

### Key Findings:
- **Random Forest Regression** performed best with an R² of 0.811 and the lowest RMSE/MAE.
- Hyperparameter tuning improved Decision Tree performance significantly (R² from 0.625 to 0.699).
- Linear Regression showed the weakest performance due to inherent simplicity.

## Conclusion
The **Random Forest Regression** model is recommended for predicting California housing prices due to its robustness and high accuracy. It effectively captures complex relationships in the data while minimizing overfitting.

Future work could explore:
- Incorporating additional features (e.g., crime rates, school quality).
- Testing advanced models like Gradient Boosting or Neural Networks.
- Deploying the model as a web service for real-time predictions.
