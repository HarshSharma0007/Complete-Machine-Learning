import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
import os
os.makedirs("models", exist_ok=True)  # Ensure the models directory exists
# data = pd.read_csv("data\\diabetes.csv")
# data = pd.read_csv("data/diabetes.csv")  # Relative path
data = pd.read_csv(r"C:\Users\DELL\Documents\GitHub\Complete-Machine-Learning\WorkingProjects\DiabetesApp\app\data\diabetes.csv")  # Use forward slashes for Linux-based systems (Docker)

X = data.drop('Outcome', axis =1)
y = data['Outcome']
# model = LogisticRegression()
model = LogisticRegression(max_iter=1000, solver='lbfgs')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model.fit(X_scaled, y)


# model.fit(X, y)
# joblib.dump(model, 'WorkingProjects\DiabetesApp\models\logreg_model.joblib')

joblib.dump(model, "models/logreg_model.joblib")  # Save inside correct folder


