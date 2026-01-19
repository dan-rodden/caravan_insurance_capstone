import pickle
from flask import Flask, request, jsonify
import xgboost as xgb

model_file = "xgb_model_eta=0.05_depth=3_min-child=6_v0.0.bin"

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# name of the flask app
app = Flask('caravan_insurance_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    # get the customer data
    customer = request.get_json()

    # transform the customer data
    X = dv.transform([customer])
    # convert to DMatrix
    dX = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
    # predict
    y_pred = model.predict(dX)
    probability = y_pred[0]

    result = {
        'probability': float(probability),
        'potential_customer': bool(probability >= 0.5)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)