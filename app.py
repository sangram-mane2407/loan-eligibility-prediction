from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("loan_model.pkl")

def get_denial_reason(features):
    reasons = []
    if features[2] < 3000:
        reasons.append("Low Applicant Income")
    if features[4] > 250:
        reasons.append("High Loan Amount")
    if features[6] == 0:
        reasons.append("No Credit History")
    return ", ".join(reasons) if reasons else "Eligible"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    reason = None
    percentage = None

    if request.method == "POST":
        try:
            features = [
                int(request.form["Gender"]),
                int(request.form["Married"]),
                int(request.form["ApplicantIncome"]),
                float(request.form["CoapplicantIncome"]),
                int(request.form["LoanAmount"]),
                int(request.form["Loan_Amount_Term"]),
                float(request.form["Credit_History"]),
                int(request.form["Education"]),
                int(request.form["Self_Employed"]),
                int(request.form["Property_Area"])
            ]

            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0][1]  # Probability of Eligible
            percentage = int(probability * 100)

            if prediction == 1:
                result = f"Eligible ✅ ({percentage}%)"
                reason = None
            else:
                result = f"Not Eligible ❌ ({percentage}%)"
                reason = get_denial_reason(features)

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template("index.html", result=result, reason=reason, percentage=percentage)

if __name__ == "__main__":
    app.run(debug=True)
