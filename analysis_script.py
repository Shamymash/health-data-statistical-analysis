import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=200):
    """
    Generates a synthetic dataset representing patient health metrics.
    Variables: Age, Physical_Activity_Score, Cognitive_Score
    """
    age = np.random.randint(50, 90, n_samples)
    
    # physical activity decreases slightly with age + random noise
    activity = 100 - (age * 0.5) + np.random.normal(0, 5, n_samples)
    
    # cognitive score is positively correlated with activity + random noise
    cognitive_score = (activity * 0.8) - (age * 0.2) + np.random.normal(0, 8, n_samples)
    
    data = pd.DataFrame({
        'Age': age,
        'Physical_Activity_Index': activity,
        'Cognitive_Test_Score': cognitive_score
    })
    return data

def analyze_data(df):
    """
    Performs exploratory data analysis and linear regression.
    """
    # 1. Exploratory Correlation
    correlation = df.corr()
    print("--- Correlation Matrix ---")
    print(correlation)
    
    # 2. Prepare for Regression
    X = df[['Age', 'Physical_Activity_Index']]
    y = df['Cognitive_Test_Score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Predict and Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n--- Model Performance ---")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    
    return model, predictions

if __name__ == "__main__":
    print("Initializing Health Data Analysis...")
    
    # Generate Data
    patient_data = generate_synthetic_data()
    print(f"Dataset generated with {len(patient_data)} samples.")
    
    # Run Analysis
    model, preds = analyze_data(patient_data)
    
    print("\nAnalysis Complete.")
