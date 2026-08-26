"""
Model evaluation results and metadata for Food Delivery Time Prediction project.
Contains verified metrics obtained during model evaluation.
"""

MODEL_RESULTS = {
    "Linear Regression": {
        "MAE": 8.49,
        "MSE": 109.92,
        "RMSE": 10.48,
        "R2": 0.65,
        "Selected": True
    },
    "Decision Tree": {
        "MAE": 11.61,
        "MSE": 209.89,
        "RMSE": 14.49,
        "R2": 0.34,
        "Selected": False
    },
    "Random Forest": {
        "MAE": 9.13,
        "MSE": 124.71,
        "RMSE": 11.17,
        "R2": 0.61,
        "Selected": False
    },
    "Gradient Boosting": {
        "MAE": 8.66,
        "MSE": 113.04,
        "RMSE": 10.63,
        "R2": 0.64,
        "Selected": False
    },
    "Tuned Random Forest": {
        "MAE": 9.22,
        "MSE": 125.25,
        "RMSE": 11.19,
        "R2": 0.60,
        "Selected": False
    }
}

PROJECT_METADATA = {
    "title": "Food Delivery Time Prediction Using Machine Learning",
    "subtitle": "Machine Learning Based Delivery Time Estimation",
    "badge": "Regression • Machine Learning • Streamlit",
    "dataset_records": "~800",
    "num_features": 10,
    "target": "Delivery_Time_min",
    "best_model": "Linear Regression",
    "best_r2": 0.65,
    "best_mae": 8.49,
    "sklearn_version": "1.6.1",
    "github_username": "Abrar-Codes404",
    "github_url": "https://github.com/Abrar-Codes404"
}

FEATURE_RANGES = {
    "Delivery_Person_Age": {"min": 18, "max": 70, "default": 28},
    "Delivery_Person_Rating": {"min": 1.0, "max": 5.0, "default": 4.5, "step": 0.1},
    "Multiple_Deliveries": {"min": 0, "max": 5, "default": 1},
    "Distance_km": {"min": 0.1, "max": 50.0, "default": 5.0, "step": 0.1},
    "Weather_options": ["Sunny", "Cloudy", "Rainy", "Stormy", "Windy"],
    "Traffic_options": ["Low", "Medium", "High", "Jam"],
    "Vehicle_options": ["Motorcycle", "Scooter", "Bike", "Bicycle", "Car"],
    "Area_options": ["Urban", "Suburban", "Rural"],
    "Order_options": ["Breakfast", "Lunch", "Dinner", "Snack"],
    "Festival_options": ["No", "Yes"]
}
