import numpy as np
from flask import Flask, request, jsonify, render_template
import sklearn.externals
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
model = joblib.load(open('crop_prediction.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('pred.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            f1 = float(request.form['f1'])
            f2 = float(request.form['f2'])
            f3 = float(request.form['f3'])
            f4 = float(request.form['f4'])

            feature_array = [f1, f2, f3, f4]
            feature = np.array(feature_array).reshape(1, -1)

            prediction = model.predict(feature)
            
            # Map prediction value to crop name
            dic = {'rice':0, 'wheat':1, 'Mung Bean':2, 'Tea':3, 'millet':4, 'maize':5, 'Lentil':6,
                   'Jute':7, 'Coffee':8, 'Cotton':9, 'Ground Nut':10, 'Peas':11, 'Rubber':12,
                   'Sugarcane':13, 'Tobacco':14, 'Kidney Beans':15, 'Moth Beans':16, 'Coconut':17,
                   'Black gram':18, 'Adzuki Beans':19, 'Pigeon Peas':20, 'Chickpea':21, 'banana':22,
                   'grapes':23, 'apple':24, 'mango':25, 'muskmelon':26, 'orange':27, 'papaya':28,
                   'pomegranate':29, 'watermelon':30}
            for key, value in dic.items():
                if value == prediction:
                    predicted_crop_name = key

            return render_template('pred.html', prediction='Prediction: {}'.format(predicted_crop_name))
        except ValueError:
            # Handle the case where non-numeric inputs are submitted
            error_message = "Please enter valid numeric values for all fields."
            return render_template('pred.html', error=error_message)


if __name__ == '__main__':
    app.run(debug=True)
    