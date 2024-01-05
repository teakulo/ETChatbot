import re

import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")

# Define patterns for greetings and broad event inquiries
greet_pattern = [{"LOWER": {"IN": ["hi", "hello", "hey", "greetings"]}}]
event_pattern = [{"LOWER": "events"}]

# Initialize the Matcher and add the patterns
matcher = Matcher(nlp.vocab)
matcher.add("GREETING", [greet_pattern])
matcher.add("EVENT_INQUIRY", [event_pattern])

def classify_intent(message):
    doc = nlp(message)
    matches = matcher(doc)

    # Check for predefined patterns first
    for match_id, start, end in matches:
        span = doc[start:end]
        intent = nlp.vocab.strings[match_id]
        return intent

    # Entity extraction for more specific intents
    if any(ent.label_ == 'GPE' for ent in doc.ents):  # Check for locations
        return 'LOCATION_INQUIRY'
    if re.search(r'\b\d+(\.\d+)?\s*BAM\b', message):  # Check for prices
        return 'PRICE_INQUIRY'
    if any(token.pos_ in ['NOUN', 'ADJ'] for token in doc):  # Check for categories/keywords
        return 'CATEGORY_KEYWORD_INQUIRY'

    return "unknown_intent"