import numpy as np
from numpy.typing import NDArray
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model


app = Flask(__name__)

model = load_model("model/LSTM-ST-Model.keras")

def predict_with_model(time_series_data: NDArray, output_len: int = 1):
    """
    Given a numpy array return the output
    Input: [X_1, X_2, X_3, ..., X_20]
    Output: [Y_1]
    """
    data_mean = time_series_data.mean()
    data_std = time_series_data.std()

    normalized_data = (time_series_data - data_mean) / data_std

    predicted_data = model.predict([[normalized_data]])
    denormalized_data = predicted_data.flatten()*data_std + data_mean
    
    return denormalized_data


@app.route("/predict", methods=["POST"])
def predict():
    """
    Given 
    """
    content = request.json
    time_series_data = content.get("data")

    if time_series_data is None:
        return "Error"

    output_len = 1
    predicted_values = predict_with_model(np.array(time_series_data), output_len=output_len)

    response = {
        "output_len": output_len,
        "prediction": predicted_values
    }
    return jsonify(response)


if __name__ == '__main__':
 
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()