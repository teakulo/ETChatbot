import random
import re
from datetime import timedelta
import dateparser
from flask import Flask, render_template, request, jsonify
import utils
from intents import classify_intent, nlp
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
events_dataset = utils.load_events_data('sample_events.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        user_message = request.form['user_message']
        intent = classify_intent(user_message)

        # Return a simple greeting message as HTML
        if intent == 'GREETING':
            return "<p>Hello! How can I help you today?</p>", 200, {'Content-Type': 'text/html'}

        # Handle specific inquiries by returning an HTML table
        elif intent == 'SPECIFIC_INQUIRY':
            extracted_entities = utils.extract_message_entities(user_message)
            matching_events = [event for event in events_dataset if utils.event_matches_criteria(event, extracted_entities)]
            response_html = utils.format_events_info(matching_events) if matching_events else "<p>No matching events found.</p>"
            return response_html, 200, {'Content-Type': 'text/html'}

        # Handle general inquiries by returning an HTML table of random events
        elif intent == 'GENERAL_INQUIRY':
            random_events = random.sample(events_dataset, min(len(events_dataset), 5))
            response_html = utils.format_events_info(random_events)
            return response_html, 200, {'Content-Type': 'text/html'}

        # Fallback for unrecognized intents
        else:
            return "<p>I'm not sure how to respond to that.</p>", 200, {'Content-Type': 'text/html'}

    except Exception as e:
        # If there is an error, return it as HTML
        return f"<p>Sorry, I encountered an error processing your request: {e}</p>", 200, {'Content-Type': 'text/html'}


def handle_specific_inquiry(user_message, events_dataset):
    extracted_entities = utils.extract_message_entities(user_message)
    matching_events = [event for event in events_dataset if utils.event_matches_criteria(event, extracted_entities)]
    response = utils.format_events_info(matching_events) if matching_events else "No matching events found."
    return jsonify({'response': response})

def handle_general_inquiry(events_dataset):
    random_events = random.sample(events_dataset, min(len(events_dataset), 5))
    response = utils.format_events_info(random_events)
    return jsonify({'response': response})

def handle_unrecognized_intent():
    return jsonify({'response': "I'm sorry, I didn't understand your query."})


if __name__ == '__main__':
    app.run(debug=True)