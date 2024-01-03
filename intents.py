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
    for match_id, start, end in matches:
        span = doc[start:end]
        intent = nlp.vocab.strings[match_id]
        return intent
    return "unknown_intent"
