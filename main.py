from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__, template_folder='C:\\Data Science\\Data Engineering\\Developpement')

## Charger le modèle entraîné
with open('C:\\Data Science\\Data Engineering\\Developpement\\rf_regressor.pkl','rb'):
    model = joblib.load("C:\\Data Science\\Data Engineering\\Developpement\\rf_regressor.pkl")

@app.route('/')
def home():
    return render_template('indexx.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('indexx.html', prediction_text='Les charges d\'assurances maladies prévues sont de {} franc.'.format(output))

if __name__ == "__main__":
    app.run(debug=True)