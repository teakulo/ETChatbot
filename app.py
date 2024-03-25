import random
from flask import Flask, render_template, request
import utils
import config  # Ensure this module is correctly set up to hold the global events_dataset
from intents import classify_intent
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        user_message = request.form['user_message']
        intent = classify_intent(user_message)

        logging.debug(f"Classified Intent: {intent}")  # Log the classified intent

        # Handle the response based on the intent
        if intent == 'GREETING':
            response_html = "<p>Hello! How can I help you today?</p>"
        elif intent in ['SPECIFIC_INQUIRY', 'GENERAL_INQUIRY']:
            extracted_entities = utils.extract_message_entities(user_message)
            if intent == 'GENERAL_INQUIRY':
                # For general inquiries, select random events
                matching_events = random.sample(config.events_dataset, min(len(config.events_dataset), 5))
            else:
                # For specific inquiries, find matching events based on extracted entities
                matching_events = [event for event in config.events_dataset if utils.event_matches_criteria(event, extracted_entities)]
            response_html = utils.format_events_info(matching_events) if matching_events else "<p>No matching events found.</p>"
        else:
            response_html = "<p>I'm not sure how to respond to that.</p>"

        return response_html, 200, {'Content-Type': 'text/html'}

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return f"<p>Sorry, I encountered an error processing your request: {e}</p>", 200, {'Content-Type': 'text/html'}

if __name__ == '__main__':
    # Load events data once at the start and store it globally in 'config.events_dataset'
    config.events_dataset = utils.load_events_data('sample_events.csv')
    app.run(debug=True)
