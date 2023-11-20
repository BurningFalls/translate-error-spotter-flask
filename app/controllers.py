import re

from app import app
from flask import jsonify, request
from .model_loader import text_classifier_korean
from .model_loader import text_classifier_english

labels_korean = ["기쁨", "분노", "불안", "슬픔"]
labels_english = ["joy", "anger", "fear", "sadness"]


def predict_sentiment(sentence, index):
    preds_list = []
    if index == 'K':
        preds_list = text_classifier_korean(sentence)[0]
    elif index == 'E':
        preds_list = text_classifier_english(sentence)[0]

    predicted_label = int(re.sub(r'[^0-9]', '', preds_list[0]['label']))
    # predicted_score = preds_list[0]['score']

    real_label = ""
    if index == 'K':
        real_label = labels_korean[predicted_label]
    elif index == 'E':
        real_label = labels_english[predicted_label]

    return real_label


@app.route('/classify/korean', methods=['POST'])
def classify_korean_text():
    if 'text' in request.json:
        text = request.json['text']
        sentiment = predict_sentiment(text, 'K')
        return jsonify({"sentiment": sentiment})
    else:
        return jsonify({'error': 'Missing text in request'}), 400


@app.route('/classify/english', methods=['POST'])
def classify_english_text():
    if 'text' in request.json:
        text = request.json['text']
        sentiment = predict_sentiment(text, 'E')
        return jsonify({"sentiment": sentiment})
    else:
        return jsonify({'error': 'Missing text in request'}), 400
