import pandas as pd
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
feature_order = joblib.load(os.path.join(BASE_DIR, "feature_order.joblib"))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form

        input_values = {
            'rain': float(form.get('rain', 0)),
            'snow': float(form.get('snow', 0)),
            'day': int(form.get('day', 0)),
            'month': int(form.get('month', 0)),
            'year': int(form.get('year', 0)),
            'hours': int(form.get('hours', 0)),
            'minutes': int(form.get('minutes', 0)),
            'seconds': int(form.get('seconds', 0)),
        }

        input_data = dict.fromkeys(feature_order, 0)
        input_data.update(input_values)

        holiday = form.get('holiday', '').strip()
        weather = form.get('weather', '').strip()

        holiday_feature = f"holiday_{holiday}"
        weather_feature = f"weather_{weather}"

        if holiday_feature in input_data:
            input_data[holiday_feature] = 1

        if weather_feature in input_data:
            input_data[weather_feature] = 1

        final_input = pd.DataFrame([input_data])
        final_input = final_input[feature_order]

        prediction = model.predict(final_input)[0]
        estimated_volume = round(float(prediction), 2)

        return render_template('result.html', prediction=estimated_volume)

    except Exception as e:
        return f"Error in prediction logic: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
