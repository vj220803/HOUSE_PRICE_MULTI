# House_Price_Prediction (Multiple Features)

A Machine Learning-based web application that predicts house prices using features such as State, City, Locality,   Property Type, BHK, Size in SqFt, Price per SqFt, Year Built, Furnished Status, Floor Number, Total Floors, Age of Property(years), Nearby Schools, Hospitals, Parking Transport Acess, Parking Space, Security, Amenities, Facing, Owner Type, Availability Status. 
This project demonstrates how to build a production-ready regression pipeline using Multiple Linear Regression, Ridge, and Lasso models enhanced with Polynomial Features.

![Features](https://github.com/vj220803/HOUSE_PRICE_MULTI/blob/main/House%20Price%20Prediction(Multiple%20Features)%20-%20visual%20selection.png)

## Objective
To build an accurate, efficient, and user-friendly House Price Prediction system that leverages real estate data to provide real-time pricing predictions. It demonstrates regression modeling, feature engineering, evaluation, and deployment using a modern web interface.


## Features
- Predicts house prices in INR (₹ Lakhs) based on user inputs.
- Supports both numerical and categorical inputs.
- Uses `Polynomial Features` for improved accuracy.
- Applies `StandardScaler` to normalize features.
- Encodes categorical data using `LabelEncoder`.
- Selects the best features using a pre-trained feature selection strategy.
- Modern and interactive Streamlit UI dashboard.


## Tech Stack
| Component           | Library/Tool          |
|---------------------|-----------------------|
| Frontend            | Streamlit             |
| Model               | Multiple_Linear_Regression, Ridge Regression, Lasso Regression|
| Data Preprocessing  | scikit-learn (Scaler, LabelEncoder, PolynomialFeatures) |
| Deployment Ready    | Streamlit Cloud / Local server |
| Model Persistence   | Joblib (`.pkl` files) |


## Project_Structure .
```
├── app.py                        # Main Streamlit app
├── model_files/
│   ├── lasso_model.pkl          # Trained Lasso model
│   ├── scaler.pkl               # StandardScaler
│   ├── encoders.pkl             # Dictionary of LabelEncoders
│   ├── poly_transformer.pkl     # PolynomialFeatures transformer
│   └── selected_features.pkl    # Feature selection list
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview
```


## Detailed Project Flow
### 1. Data Loading and Preprocessing
- Imported data using pandas
- Categorical columns encoded using Label Encoding
- Applied StandardScaler to normalize numerical features for improved model performance
- Checked multicollinearity and correlation matrix for insights

### 2. Feature Engineering
- Used PolynomialFeatures (degree = 2) to generate new interaction terms and higher-order features
- Captured non-linear patterns between variables and improved predictive accuracy

### 3. Model Training
- Trained and saved three models:
- Multiple Linear Regression – baseline model
- Ridge Regression – uses L2 regularization to reduce overfitting
- Lasso Regression – uses L1 regularization to eliminate irrelevant features
- Each model was trained on the preprocessed and polynomial-transformed dataset. Final models were saved using joblib in the model_file/ folder.

### 4. Model Evaluation
Evaluated using:
- MAE: Mean Absolute Error
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- R² Score: Measures overall model fit
- Ridge and Lasso showed improved generalization and reduced error over standard regression.

### 5. Web App Deployment
- Built a user-friendly Streamlit interface
- Takes user input (area, bedrooms, etc.) from sidebar form
- Applies saved StandardScaler and PolynomialFeatures
- Predicts using saved Multiple Linear Regression model
- Displays real-time house price prediction in INR

### 6. Model Evaluation 

| Model              | MAE ↓   | MSE ↓   | RMSE ↓  | R² Score ↑ |
|--------------------|---------|---------|---------|-------------|
| Linear Regression  | Higher  | Higher  | Higher  | Lower       |
| Ridge Regression   | Lower   | Lower   | Lower   | Higher      |
| Lasso Regression   | Lower   | Lower   | Lower   | Higher      |




## How to Run the App Locally

### 1. Clone the Repository
```
git clone https://github.com/your-username/house-price-predictor.git
cd house-price-predictor
```

### 2. Create and Activate a Virtual Environment(Optional but Recommended)
```
For macOS/Linux
python -m venv venv
source venv/bin/activate
```
```
For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```
streamlit run app.py
```


## Dependencies
- streamlit
- pandas
- numpy
- scikit-learn
- joblib


## Key Concepts Demonstrated
- Multiple Regression Techniques
- Polynomial Feature Transformation
- Regularization (Ridge & Lasso)
- Model Evaluation Metrics
- Scikit-learn Pipelines
- Web App Deployment using Streamlit
- .pkl Model Serialization for Real-time Use

## Final UI Dashboard
![Dashboard](https://github.com/vj220803/HOUSE_PRICE_MULTI/blob/main/HOUSI_MULTI_UI.png)

## Author
Vijayan Naidu 
Student @ Fergusson College(Autonomous) Pune, Maharashtra


