import random
import re
from datetime import timedelta

import dateparser
import numpy as np
from flask import Flask, render_template, request
from sklearn.neighbors import NearestNeighbors
from flask import jsonify
import utils
from intents import classify_intent, nlp

app = Flask(__name__)

events_dataset = utils.load_events_data('sample_events.csv')

# Set up KNN
#knn = NearestNeighbors(n_neighbors=5)
#knn.fit(utils.features)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        user_message = request.form['user_message']
        print(f"Received user message: {user_message}")  # Debugging
        intent = classify_intent(user_message)
        print(f"Identified intent: {intent}")  # Debugging

        if intent == 'GREETING':
            return jsonify({'response': "Hello! How can I help you today?"})

        if intent == 'LOCATION_INQUIRY':
            return handle_location_inquiry(user_message)

        if intent == 'PRICE_INQUIRY':
            return handle_price_inquiry(user_message)

        if intent == 'CATEGORY_KEYWORD_INQUIRY':
            return handle_category_keyword_inquiry(user_message, events_dataset)

        # For unrecognized intents
        return handle_unrecognized_intent()

    except Exception as e:
        print(f"An error occurred in get_response: {e}")  # Debugging
        return jsonify({'response': "Sorry, I encountered an error processing your request."})

def handle_location_inquiry(user_message):
    doc = nlp(user_message)
    locations = [ent.text for ent in doc.ents if ent.label_ == 'GPE']

    if locations:
        response = f"Sure, here are some events in {', '.join(locations)}:"
    else:
        response = "I'm sorry, I couldn't detect a location in your inquiry."

    return jsonify({'response': response})
def handle_price_inquiry(user_message):
    price_pattern = r'\b\d+(\.\d+)?\s*BAM\b'
    prices = re.findall(price_pattern, user_message)

    if prices:
        response = f"Sure, here are some events within your budget of {', '.join(prices)} BAM:"
    else:
        response = "I'm sorry, I couldn't find any prices in your inquiry."

    return jsonify({'response': response})


def handle_category_keyword_inquiry(user_message, events_dataset):
    # Extract entities and keywords from the message
    extracted_entities = utils.extract_message_entities(user_message)

    matching_events = []

    # Iterate through the events dataset
    for event in events_dataset:
        # Check if the event matches the extracted criteria
        if utils.event_matches_criteria(event, extracted_entities):
            matching_events.append(event['name'])  # Add the event name to the list
        # Determine the time frame from the user's message
    time_frame = get_time_frame(user_message)

    matching_events = []

    for event in events_dataset:
        event_date = dateparser.parse(event.get('date'))  # Assuming each event has a 'date' key
        # Check if the event date falls within the time frame
        if time_frame and event_date and time_frame <= event_date <= time_frame + timedelta(days=1):
            if utils.event_matches_criteria(event, extracted_entities):
                matching_events.append(event['name'])

    if matching_events:
        response = f"Here are some events that match your query: {', '.join(matching_events)}."
    else:
        response = "I couldn't find any events matching your query."

    return jsonify({'response': response})

    # Construct the response based on the matching events
    if matching_events:
        response = f"Here are some events that match your query: {', '.join(matching_events)}."
    else:
        response = "I couldn't find any events that match your query."

    return jsonify({'response': response})



def handle_unrecognized_intent():
    # Inform the user that the query wasn't understood
    return jsonify({'response': "I'm sorry, I didn't understand your query."})

def handle_event_inquiry(user_message):
    extracted_entities = utils.extract_message_entities(user_message)
    print(f"Extracted entities: {extracted_entities}")  # Debugging

    # Filter events based on extracted entities
    filtered_events = [event for event in events_dataset if utils.event_matches_criteria(event, extracted_entities)]
    print(f"Filtered events: {filtered_events}")  # Debugging

    if not filtered_events:  # No specific criteria or no matching events
        recommended_events = random.sample(events_dataset, 5) if len(events_dataset) > 5 else events_dataset
    else:
        recommended_events = filtered_events[:5]

    event_names = [event['name'] for event in recommended_events if 'name' in event]
    return jsonify({'events': event_names})


if __name__ == '__main__':
    app.run(debug=True)
