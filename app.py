from flask import Flask, jsonify
import pandas as pd

# Load the CSV dataset into a pandas DataFrame
df = pd.read_csv("studentInfo.csv")
print(df.loc[[0, 1, 2]])

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

if __name__ == "__main__":
    app.run(debug=True)
