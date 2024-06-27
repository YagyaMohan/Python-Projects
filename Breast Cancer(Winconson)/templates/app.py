import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from flask import Flask, render_template, request

app = Flask(__name__)

def load_model():
    data = pd.read_csv(r"C:\Users\mohan\OneDrive\Desktop\DS and ML\Breast Cancer(Winconson) Project\data.csv")
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    if 'id' in data.columns:
        data = data.drop(columns=['id'])
    if 'Unnamed: 32' in data.columns:
        data = data.drop(columns=['Unnamed: 32'])  # Drop Unnamed: 32 column if it exists

    feature_cols = data.columns.drop('diagnosis')
    X = data[feature_cols]
    y = data['diagnosis']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    return model, feature_cols

model, feature_cols = load_model()

@app.route('/')
def home():
    return render_template('breast_cancer_form.html', feature_cols=feature_cols)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            input_data = {col: float(request.form[col]) for col in feature_cols if col != 'Unnamed: 32'}
            input_df = pd.DataFrame([input_data], columns=feature_cols)
            prediction = model.predict(input_df)
            result = 'Malignant' if prediction[0] == 1 else 'Benign'

            return render_template('result.html', result=result)
        except ValueError as e:
            return render_template('error.html', message=f'Invalid input values: {e}')

if __name__ == '__main__':
    app.run(debug=True)
