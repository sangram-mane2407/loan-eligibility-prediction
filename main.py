import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the training dataset
data = pd.read_csv("loan.csv")

# Fill missing values
data.ffill(inplace=True)  # ✅ Fixed FutureWarning

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# ✅ Use all 10 form features
X = data[['Gender', 'Married', 'ApplicantIncome', 'CoapplicantIncome',
          'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
          'Education', 'Self_Employed', 'Property_Area']]

y = data['Loan_Status']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'loan_model.pkl')
print("✅ Model trained and saved as loan_model.pkl")
