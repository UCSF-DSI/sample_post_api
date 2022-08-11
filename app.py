import json


import logging

from flask import Flask, request

import fasttext

# create the Flask app
app = Flask(__name__)

# load text classifier
model = fasttext.load_model("clf_model.bin")

@app.route('/get_example/', methods = ['GET'])
def get_example():
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    user = request.args.get('user')
    query = request.args.get('query')
    logging.info(f"User Query: {query}")
    
    model_prediction = model.predict(query, k = 1)
    
    results = {
        'user': user,
        'query': query,
        'predicted_intent': model_prediction[0][0],
        'prediction_score': model_prediction[1][0]
    }
    
    results['response'] = "I'd be happy to help you get started!" \
        if model_prediction[0][0] == "__label__getting_started" \
        else "I'd be happy to tell you more about the features"
    
    return json.dumps(results)

@app.route('/post_example/', methods = ['POST'])
def query_example():
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    results = request.get_json()
    logging.info(f"User Query: {results['query']}")
    
    model_prediction = model.predict(results['query'], k = 1)
    
    results['predicted_intent'] = model_prediction[0][0]
    results['prediction_score'] = model_prediction[1][0]
    
    results['response'] = "I'd be happy to help you get started!" \
        if model_prediction[0][0] == "__label__getting_started" \
        else "I'd be happy to tell you more about the features"
    
    return json.dumps(results)

if __name__ == '__main__':
    app.run()
