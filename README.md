# House_Price_Prediction (Multiple Features)

A Machine Learning-based web application that predicts house prices using features such as State, City, Locality,   Property Type, BHK, Size in SqFt, Price per SqFt, Year Built, Furnished Status, Floor Number, Total Floors, Age of Property(years), Nearby Schools, Hospitals, Parking Transport Acess, Parking Space, Security, Amenities, Facing, Owner Type, Availability Status. 
This project demonstrates how to build a production-ready regression pipeline using Multiple Linear Regression, Ridge, and Lasso models enhanced with Polynomial Features.


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

## How It Works
### 1. User Inputs: 
Users enter property information (location, size, amenities, etc.).
### 2. Preprocessing: 
- Categorical features are encoded using saved LabelEncoders.
- Numerical features are scaled using the StandardScaler.
- Polynomial features are generated.
### 3.Prediction:
- The processed input is passed to a trained Lasso model.
- The model outputs the predicted price in lakhs.


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






