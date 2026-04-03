from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scale = pickle.load(open('encoder.pkl', 'rb'))

names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day',
         'hours', 'minutes', 'seconds']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form = request.form

    input_features = [
        float(form.get('holiday', 0)),
        float(form.get('temp', 0)),
        float(form.get('rain', 0)),
        float(form.get('snow', 0)),
        float(form.get('weather', 0)),
        float(form.get('year', 0)),
        float(form.get('month', 0)),
        float(form.get('day', 0)),
        float(form.get('hours', 0)),
        float(form.get('minutes', 0)),
        float(form.get('seconds', 0))
    ]

    features_values = np.array([input_features])
    data = pd.DataFrame(features_values, columns=names)

    scaled_data = scale.transform(data)

    prediction = model.predict(scaled_data)

    text = "Estimated Traffic Volume is: "
    return render_template('result.html', prediction=text + str(prediction[0]))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True)
