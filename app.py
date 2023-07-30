from flask import Flask, jsonify, request
import pandas as pd

# Perform any data processing or cleaning if required
# For example:
# df = df.dropna()  # Drop rows with missing values
# df = df.reset_index(drop=True)  # Reset the DataFrame index

app = Flask(__name__)

@app.route("/api/data", methods=["GET"])
def get_all_data():
    csv_file_path = request.headers.get('file-path')
    
    if csv_file_path is None:
        return jsonify({"message": "File path must be provided as an argument"}), 400

    # Load the CSV dataset into a pandas DataFrame
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        return jsonify({"message": "File not found"}), 404

    return jsonify(df.to_dict(orient="records"))

@app.route("/api/data/<int:id>", methods=["GET"])
def get_data_by_id(id):
    # Find data by ID and return as JSON
    data = df[df["id"] == id].to_dict(orient="records")
    return jsonify(data[0]) if data else jsonify({"message": "Data not found"})

if __name__ == "__main__":
    app.run(debug=True)
