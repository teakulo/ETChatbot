import numpy as np
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from flask import jsonify
import utils
from config import categorical_features

app = Flask(__name__)

events_dataset = utils.load_events_data('sample_events.csv')

# Set up KNN
knn = NearestNeighbors(n_neighbors=5)
knn.fit(utils.features)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/get_response', methods=['POST'])
# app.py, within the get_response function

def get_response():
    try:
        user_message = request.form['user_message']
        query_type = utils.classify_query(user_message)

        # Debugging snippet to check the type of each event in events_dataset
        for event in events_dataset:
            if not isinstance(event, dict):
                print(f"Expected a dict, but got a {type(event)}: {event}")

        # You only need to check for event keywords if the query is about event recommendation
        if query_type == 'event_recommendation':
            parsed_date = utils.parse_date(user_message)
            if parsed_date:
                # If a date is parsed successfully, look for events on that date
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
                # If no date is parsed, proceed with keyword extraction and recommendation
                event_keywords = utils.extract_keywords_from_descriptions(
                    [event['description'] for event in events_dataset]
                )
                genre_keywords = utils.extract_genre_keywords(events_dataset)
                encoded_query = utils.encode_query(user_message, event_keywords, genre_keywords)

                # Check if the encoding was successful before proceeding
                if isinstance(encoded_query, np.ndarray):
                    recommended_events = utils.recommend_events_knn(encoded_query)
                    events_info = utils.format_events_info(recommended_events)
                    return jsonify({'events': events_info})
                else:
                    return jsonify({'response': "No matching events found."})

        elif query_type == 'help':
            # Help response
            return jsonify({
                               'response': "You can ask me about events, locations, and times. For example, 'What events are happening in Sarajevo next weekend?'"})
        else:
            # General response
            return jsonify({'response': "Hi there! How can I assist you with event information today?"})


    except Exception as e:
        # Log the exception and return a user-friendly message
        print(f"An error occurred in get_response: {e}")
        return jsonify({'response': "Sorry, I encountered an error processing your request."})


if __name__ == '__main__':
    app.run(debug=True)
