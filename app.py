from flask import Flask, jsonify, request
import pandas as pd
import joblib
import numpy as np
import pickle
import openai
import configparser

# Load the CSV dataset into a pandas DataFrame
df = pd.read_csv("./datasets/studentInfo.csv")

# Perform any data processing or cleaning if required
# For example:
# df = df.dropna()  # Drop rows with missing values
# df = df.reset_index(drop=True)  # Reset the DataFrame index

app = Flask(__name__)

@app.route("/api/data", methods=["GET"])
def get_all_data():
    # Convert DataFrame to JSON and return
    return jsonify(df.to_dict(orient="records"))

@app.route("/api/data/<int:id>", methods=["GET"])
def get_data_by_id(id):
    # Find data by ID and return as JSON
    data = df[df["id"] == id].to_dict(orient="records")
    return jsonify(data[0]) if data else jsonify({"message": "Data not found"})

# Load the model from the file
model = joblib.load('model_2.pkl')
le = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    predict_request = [data['HOURS_DATASCIENCE'], data['HOURS_BACKEND'], data['HOURS_FRONTEND'], data['NUM_COURSES_BEGINNER_DATASCIENCE'], data['NUM_COURSES_BEGINNER_BACKEND'], data['NUM_COURSES_BEGINNER_FRONTEND'], data['NUM_COURSES_ADVANCED_DATASCIENCE'], data['NUM_COURSES_ADVANCED_BACKEND'], data['NUM_COURSES_ADVANCED_FRONTEND'], data['AVG_SCORE_DATASCIENCE'], data['AVG_SCORE_BACKEND'], data['AVG_SCORE_FRONTEND']]
    predict_request = np.array(predict_request).reshape(1, -1)
    y_hat = model.predict(predict_request)
    output = le.inverse_transform(y_hat)
    print(output)
    return jsonify(results=output)

@app.route('/gpt', methods=['POST'])
def gpt():
    data = request.get_json(force=True)
    # Load OpenAI key from config file
    config = configparser.ConfigParser()
    config.read('config.ini')

    openai.api_key = config['OpenAI']['key']
    prompt_text = "Can you show me what I must do to mastered {data['prediction']} " + "as I am Backend developper"
    response = openai.Completion.create(engine='text-davinci-003', prompt=prompt_text, max_tokens=1000)
    answer = response.choices[0].text.strip()
    return jsonify(answer=answer)
 

if __name__ == "__main__":
    app.run(debug=True)
