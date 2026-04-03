import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.joblib")
feature_order = joblib.load("feature_order.joblib")

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

        input_data = {col: 0 for col in feature_order}
        input_data.update(input_values)

        holiday_feature = f"holiday_{form.get('holiday', '')}"
        weather_feature = f"weather_{form.get('weather', '')}"

        if holiday_feature in input_data:
            input_data[holiday_feature] = 1
        if weather_feature in input_data:
            input_data[weather_feature] = 1

        final_input = pd.DataFrame([input_data])

        prediction = model.predict(final_input)[0]
        estimated_volume = round(prediction, 2)

        return render_template('result.html', prediction=estimated_volume)

    except Exception as e:
        return f"Error in prediction logic: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
