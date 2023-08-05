import joblib
import numpy as np
import pandas as pd
import pickle
import openai

# Load the model from the file
model = joblib.load('model_2.pkl')
le = joblib.load('label_encoder.pkl')

# Define a sample observation with 13 features (replace this with your actual data)
feature_names = [
    "HOURS_DATASCIENCE", "HOURS_BACKEND", "HOURS_FRONTEND",
    "NUM_COURSES_BEGINNER_DATASCIENCE", "NUM_COURSES_BEGINNER_BACKEND",
    "NUM_COURSES_BEGINNER_FRONTEND", "NUM_COURSES_ADVANCED_DATASCIENCE",
    "NUM_COURSES_ADVANCED_BACKEND", "NUM_COURSES_ADVANCED_FRONTEND",
    "AVG_SCORE_DATASCIENCE", "AVG_SCORE_BACKEND", "AVG_SCORE_FRONTEND"
]

sample_data = [
    28, # HOURS_DATASCIENCE
    7,       # HOURS_BACKEND
    39,      # HOURS_FRONTEND
    29,      # NUM_COURSES_BEGINNER_DATASCIENCE
    2,       # NUM_COURSES_BEGINNER_BACKEND
    4,       # NUM_COURSES_BEGINNER_FRONTEND
    0,       # NUM_COURSES_ADVANCED_DATASCIENCE
    2,       # NUM_COURSES_ADVANCED_BACKEND
    5,       # NUM_COURSES_ADVANCED_FRONTEND
    84,      # AVG_SCORE_DATASCIENCE
    74,      # AVG_SCORE_BACKEND
    0        # AVG_SCORE_FRONTEND
]

# Make sure the input data is in the correct shape (1, number_of_features)
# input_data = np.array(sample_data).reshape(1, -1)

# Use the model to predict the target
# prediction = model.predict(input_data)

# Create a DataFrame with the correct feature names
input_data_df = pd.DataFrame([sample_data], columns=feature_names)

# Predict using the DataFrame
prediction = model.predict(input_data_df)
data = le.inverse_transform(prediction)

# Print the prediction
print("Prediction:", data)

# Set up your OpenAI API key
# API chatGPT4
openai.api_key = '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'

# Modify this prompt to be relevant to the predicted profile
prompt_text = f"Can you tell me about a student with the following profile: {data}?"

# Call the GPT API
response = openai.Completion.create(
    engine='text-davinci-003',
    prompt=prompt_text,
    max_tokens=1000
)

# Process the response
answer = response.choices[0].text.strip()

# Print the response
print(answer)