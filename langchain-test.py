import joblib
import numpy as np
import pandas as pd

from langchain.llms import OpenAI

feature_names = [
    'HOURS_DATASCIENCE',
    'HOURS_BACKEND',
    'HOURS_FRONTEND',
    'HOURS_IA',
    'HOURS_BDD',
    'NUM_COURSES_BEGINNER_DATASCIENCE',
    'NUM_COURSES_BEGINNER_BACKEND',
    'NUM_COURSES_BEGINNER_FRONTEND',
    'NUM_COURSES_BEGINNER_IA',
    'NUM_COURSES_BEGINNER_BDD',
    'NUM_COURSES_ADVANCED_DATASCIENCE',
    'NUM_COURSES_ADVANCED_BACKEND',
    'NUM_COURSES_ADVANCED_FRONTEND',
    'NUM_COURSES_ADVANCED_IA',
    'NUM_COURSES_ADVANCED_BDD',
    'AVG_SCORE_DATASCIENCE',
    'AVG_SCORE_BACKEND',
    'AVG_SCORE_FRONTEND',
    'AVG_SCORE_IA',
    'AVG_SCORE_BDD',
    'NB_CLICKS_DATASCIENCE',
    'NB_CLICKS_BACKEND',
    'NB_CLICKS_FRONTEND',
    'NB_CLICKS_IA',
    'NB_CLICKS_BDD',
    'ORIENTATION']


data_sample = [
    10,   # HOURS_DATASCIENCE
    5,    # HOURS_BACKEND
    8,    # HOURS_FRONTEND
    7,    # HOURS_IA
    9,    # HOURS_BDD
    2,    # NUM_COURSES_BEGINNER_DATASCIENCE
    1,    # NUM_COURSES_BEGINNER_BACKEND
    2,    # NUM_COURSES_BEGINNER_FRONTEND
    1,    # NUM_COURSES_BEGINNER_IA
    2,    # NUM_COURSES_BEGINNER_BDD
    1,    # NUM_COURSES_ADVANCED_DATASCIENCE
    0,    # NUM_COURSES_ADVANCED_BACKEND
    1,    # NUM_COURSES_ADVANCED_FRONTEND
    2,    # NUM_COURSES_ADVANCED_IA
    0,    # NUM_COURSES_ADVANCED_BDD
    80,   # AVG_SCORE_DATASCIENCE
    75,   # AVG_SCORE_BACKEND
    85,   # AVG_SCORE_FRONTEND
    70,   # AVG_SCORE_IA
    90,   # AVG_SCORE_BDD
    15,   # NB_CLICKS_DATASCIENCE
    5,    # NB_CLICKS_BACKEND
    8,    # NB_CLICKS_FRONTEND
    6,    # NB_CLICKS_IA
    10,   # NB_CLICKS_BDD
    1 # ORIENTATION
]

# Vérification de la longueur
assert len(data_sample) == len(feature_names)  # Ceci doit passer sans erreur



model = joblib.load('multi_target_model_4.pkl')
#le = joblib.load('label_encoder4.pkl')

input_data_df = pd.DataFrame([data_sample], columns=feature_names)
#input_data_df = pd.DataFrame(new_observation_2, columns=feature_names)


# Predict using the DataFrame
prediction = model.predict(input_data_df)
#data = le.inverse_transform(prediction)

# Create a LangChain client
llm = OpenAI(openai_api_key="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")

promptFront = "Comment generer une base de donnee sur MYSQL "

# Generate text
text = llm.predict(promptFront 
                   + "et adapte les reponses selon mon niveau sur ces 5 matieres " 
                   + prediction[0][0] 
                   + ","
                   + prediction[0][1]
                   + ","
                   + prediction[0][2]
                   + ","
                   + prediction[0][3]
                   + ","
                   + prediction[0][4])

print(promptFront 
                   + "sachant que voici mon profil de cursus scolaire " 
                   + prediction[0][0] 
                   + ","
                   + prediction[0][1]
                   + ","
                   + prediction[0][2]
                   + ","
                   + prediction[0][3]
                   + ","
                   + prediction[0][4])

# Print the text
print(text)