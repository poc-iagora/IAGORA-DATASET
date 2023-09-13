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

import llm as llm
import prediction as pred

app = Flask(__name__)

@app.route("/llm", methods=["POST"])
def index():
    requete = request.get_json()
    print(requete['q1'])
    print(requete['url'])
    #text = request.json["text"]
    result = llm.callLlm(requete['q1'],requete['url'])
    return jsonify(result)


@app.route("/prediction", methods=["POST"])
def prediction():
    question = request.get_json()
    result = pred.callPrediction(question['q2'])
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)