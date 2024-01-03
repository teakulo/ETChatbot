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
df = pd.DataFrame(events_dataset)

# One-hot encode categorical features
one_hot = OneHotEncoder()
encoded_categorical = one_hot.fit_transform(df[categorical_features])

# When fitting OneHotEncoder
df = pd.DataFrame(events_dataset)  # Your dataframe
df[categorical_features] = df[categorical_features].fillna('unknown')  # Replace NaN with 'unknown'
one_hot = OneHotEncoder(handle_unknown='ignore')  # Instantiate OneHotEncoder with handle_unknown parameter
one_hot.fit(df[categorical_features])  # Fit to the dataframe including the 'unknown' values

# TF-IDF for description
vectorizer = TfidfVectorizer(max_features=50)  # Limiting number of features for simplicity
tfidf_description = vectorizer.fit_transform(df['description'])

# Combine all features
features = np.hstack((encoded_categorical.toarray(), tfidf_description.toarray()))

# Set up KNN
knn = NearestNeighbors(n_neighbors=5)
knn.fit(features)

@app.route('/')
def index():
    return render_template('index.html')

def extract_entities(message):
    return utils.extract_entities(message, utils.nlp)

def recommend_events_knn(encoded_query):
    distances, indices = knn.kneighbors([encoded_query])
    recommended_event_indices = indices[0]
    return [df.iloc[i]['name'] for i in recommended_event_indices]

genre_keywords = utils.extract_genre_keywords(events_dataset)

@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        user_message = request.form['user_message']
        query_type = utils.classify_query(user_message)

        if query_type == 'event_recommendation':
            parsed_date = utils.parse_date(user_message)

            if parsed_date:
                formatted_date = parsed_date.strftime('%Y-%m-%d')
                recommended_events = [event for event in events_dataset if isinstance(event, dict) and formatted_date in event.get('start_time', '')]
                if recommended_events:
                    events_info = [{'name': event.get('name', 'N/A'), 'city': event.get('city', 'N/A'), 'genre': event.get('genre', 'N/A'), 'price': event.get('price', 'N/A')} for event in recommended_events]
                    return jsonify({'events': events_info})
                else:
                    return jsonify({'response': "No events found around that date."})
            else:
                event_keywords = utils.extract_keywords_from_descriptions([event['description'] for event in events_dataset])
                genre_keywords = utils.extract_genre_keywords(events_dataset)
                encoded_query = utils.encode_query(user_message, event_keywords, genre_keywords, one_hot, vectorizer, categorical_features)

                if encoded_query is not None:
                    recommended_events = recommend_events_knn(encoded_query)
                    if recommended_events:
                        events_info = [{'name': event['name'], 'city': event['city'], 'genre': event['genre'], 'price': event['price']} for event in recommended_events]
                        return jsonify({'events': events_info})
                    else:
                        return jsonify({'response': "No matching events found."})
        elif query_type == 'help':
            return jsonify({'response': "You can ask me about events, locations, and times. For example, 'What events are happening in Sarajevo next weekend?'"})
        else:
            return jsonify({'response': "Hi there! How can I assist you with event information today?"})

    except Exception as e:
        print(f"An error occurred in get_response: {e}")
        return jsonify({'response': "Sorry, I encountered an error processing your request."})

if __name__ == '__main__':
    app.run(debug=True)
