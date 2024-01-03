import numpy as np
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from flask import jsonify
import utils
from config import categorical_features
from intents import classify_intent

app = Flask(__name__)

events_dataset = utils.load_events_data('sample_events.csv')

# Set up KNN
knn = NearestNeighbors(n_neighbors=5)
knn.fit(utils.features)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        user_message = request.form['user_message']
        query_type = utils.classify_query(user_message)
        intent = classify_intent(user_message)

        # Based on the intent, you would have different logic
        if intent == 'GREETING':
            return jsonify({'response': "Hello! How can I help you today?"})
        elif intent == 'EVENT_INQUIRY':
            # Assuming you want to return all events for an event inquiry
            all_events_info = utils.format_events_info(events_dataset)
            return jsonify({'events': all_events_info})

        # You only need to check for event keywords if the query is about event recommendation
        if query_type == 'event_recommendation':
            parsed_date = utils.parse_date(user_message)
            if parsed_date:
                formatted_date = parsed_date.strftime('%Y-%m-%d')
                recommended_events = [
                    event for event in events_dataset
                    if isinstance(event, dict) and formatted_date in event.get('start_time', '')
                ]
                if recommended_events:
                    events_info = utils.format_events_info(recommended_events)
                    return jsonify({'events': events_info})
                else:
                    return jsonify({'response': "No events found around that date."})
            else:
                event_keywords = utils.extract_keywords_from_descriptions(
                    [event['description'] for event in events_dataset]
                )
                genre_keywords = utils.extract_genre_keywords(events_dataset)
                encoded_query = utils.encode_query(user_message, event_keywords, genre_keywords)

                if isinstance(encoded_query, np.ndarray):
                    recommended_events = utils.recommend_events_knn(encoded_query)
                    events_info = utils.format_events_info(recommended_events)
                    return jsonify({'events': events_info})
                else:
                    return jsonify({'response': "No matching events found."})

        elif query_type == 'help':
            return jsonify({
                'response': "You can ask me about events, locations, and times. For example, 'What events are happening in Sarajevo next weekend?'"
            })
        else:
            return jsonify({'response': "Hi there! How can I assist you with event information today?"})

    except Exception as e:
        # Log the exception and return a user-friendly message
        print(f"An error occurred in get_response: {e}")
        return jsonify({'response': "Sorry, I encountered an error processing your request."})

if __name__ == '__main__':
    app.run(debug=True)
