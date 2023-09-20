import requests
from bs4 import BeautifulSoup
import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain import PromptTemplate
import openai

import scrapUrl as su
import scrapPdf as sp

import json
from flask import Flask, request
from flask import jsonify

from flask import Flask
from flask_cors import CORS

import llm as llm
import prediction as pred
from flask import Response


app = Flask(__name__)
CORS(app)

@app.route("/llm", methods=["POST"])
def index():
    requete = request.get_json()
    print(requete['q1'])
    print(requete['url'])
    #text = request.json["text"]
    result = llm.callLlm(requete['q1'],requete['url'])
    resp = Response(result)
    resp.charset = "utf-8"
    return resp


@app.route("/prediction", methods=["POST"])
def prediction():
    question = request.get_json()
    result = pred.callPrediction(question['q2'],
        question['HOURS_DATASCIENCE'], question['HOURS_BACKEND'], question['HOURS_FRONTEND'], question['HOURS_IA'], question['HOURS_BDD'],
        question['NUM_COURSES_BEGINNER_DATASCIENCE'], question['NUM_COURSES_BEGINNER_BACKEND'], question['NUM_COURSES_BEGINNER_FRONTEND'], question['NUM_COURSES_BEGINNER_IA'], question['NUM_COURSES_BEGINNER_BDD'],
        question['NUM_COURSES_ADVANCED_DATASCIENCE'], question['NUM_COURSES_ADVANCED_BACKEND'], question['NUM_COURSES_ADVANCED_FRONTEND'], question['NUM_COURSES_ADVANCED_IA'], question['NUM_COURSES_ADVANCED_BDD'],
        question['AVG_SCORE_DATASCIENCE'], question['AVG_SCORE_BACKEND'], question['AVG_SCORE_FRONTEND'], question['AVG_SCORE_IA'], question['AVG_SCORE_BDD'],
        question['ORIENTATION'],
        question['FAV_COURS'], question['HAT_COURS'])
    resp = Response(result)
    resp.charset = "utf-8"
    return resp
