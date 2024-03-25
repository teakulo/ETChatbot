import spacy
from spacy.matcher import Matcher
import logging
from utils import load_events_data, extract_message_entities  # Assuming these functions are defined in utils.py

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

def update_matcher_with_dataset(events_dataset):
    """Updates the matcher with patterns derived from the events dataset."""
    for event in events_dataset:
        # Extract and add unique keywords for categories, venues, cities, etc.
        for field in ['category', 'venue', 'city', 'name', 'description']:
            if event[field]:
                # Split categories and add each as a pattern
                for value in event[field].split(','):
                    pattern = [{"LOWER": value.lower().strip()}]
                    matcher.add(field.upper(), [pattern])

def classify_dynamic_intent(message):
    """Classifies intent dynamically based on updated matcher patterns."""
    doc = nlp(message.lower())
    matches = matcher(doc)

    # Debug: log matched entities
    for match_id, start, end in matches:
        span = doc[start:end]
        logging.debug(f"Matched: {span.text} for pattern: {nlp.vocab.strings[match_id]}")

    if matches:
        return 'SPECIFIC_INQUIRY'
    else:
        if message.strip().lower() == "events?":
            return 'GENERAL_INQUIRY'
        extracted_entities = extract_message_entities(message)
        if extracted_entities['keywords']:
            return 'SPECIFIC_INQUIRY'
        return 'GENERAL_INQUIRY'

# Load events data and update matcher patterns
events_dataset = load_events_data('sample_events.csv')
update_matcher_with_dataset(events_dataset)


