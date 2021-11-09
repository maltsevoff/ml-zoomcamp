import pickle
from flask import Flask
from flask import request
from flask import jsonify

# ### Load the model

model_file = 'model_C=1.0.bin'
app = Flask('churn')

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    probability = model.predict_proba(X)[0, 1]
    churn = probability >= 0.5

    result = {
        'churn_probability': float(probability),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
