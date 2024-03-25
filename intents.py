import spacy
from spacy.matcher import Matcher

from config import events_dataset
from dynamic_intent_classifier import classify_dynamic_intent
from utils import extract_message_entities

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Define patterns for greetings
greet_pattern = [{"LOWER": {"IN": ["hi", "hello", "hey", "greetings"]}}]
matcher.add("GREETING", [greet_pattern])

def classify_intent(message):
    doc = nlp(message)
    matches = matcher(doc)

    # Check for predefined greeting patterns first
    for match_id, start, end in matches:
        span = doc[start:end]
        return nlp.vocab.strings[match_id]

    if message.strip().lower() == "events?":
        return 'GENERAL_INQUIRY'

        # Delegate to the dynamic classifier for specific inquiries
    extracted_entities = extract_message_entities(message)
    dynamic_intent = classify_dynamic_intent(message)
    if dynamic_intent == 'SPECIFIC_INQUIRY':
        return 'SPECIFIC_INQUIRY'
    else:
        return 'GENERAL_INQUIRY'