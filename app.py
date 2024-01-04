import random
import numpy as np
from flask import Flask, render_template, request
from sklearn.neighbors import NearestNeighbors
from flask import jsonify
import utils
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
        intent = classify_intent(user_message)

        if intent == 'GREETING':
            return jsonify({'response': "Hello! How can I help you today?"})
        # Handle the event inquiry intent
        elif intent == 'EVENT_INQUIRY':
            # Select 5 random events from the dataset
            if len(events_dataset) > 5:
                recommended_events = random.sample(events_dataset, 5)
            else:
                recommended_events = events_dataset

            # Respond with the names of the selected events
            event_names = [event['name'] for event in recommended_events if 'name' in event]
            return jsonify({'events': event_names})
        else:
            # Handle other intents based on the classified query type
            query_type = utils.classify_query(user_message)
            if query_type == 'event_recommendation':
                # Process event recommendation
                parsed_date = utils.parse_date(user_message)
                if parsed_date:
                    formatted_date = parsed_date.strftime('%Y-%m-%d')
                    recommended_events = [
                        event for event in events_dataset
                        if isinstance(event, dict) and formatted_date in event.get('start_time', '')
                    ]
                    if recommended_events:
                        events_info = [{'name': event['name']} for event in recommended_events]
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
                        recommended_events_indices = knn.kneighbors([encoded_query], return_distance=False)
                        recommended_events = [events_dataset[i] for i in recommended_events_indices[0]]
                        events_info = [{'name': event['name']} for event in recommended_events]
                        return jsonify({'events': events_info})
                    else:
                        return jsonify({'response': "No matching events found."})

            elif query_type == 'help':
                return jsonify({
                    'response': "You can ask me about events, locations, and times."
                })
            else:
                return jsonify({'response': "Hi there! How can I assist you with event information today?"})

    except Exception as e:
        print(f"An error occurred in get_response: {e}")
        return jsonify({'response': "Sorry, I encountered an error processing your request."})

if __name__ == '__main__':
    app.run(debug=True)
