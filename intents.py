import re
import spacy
from spacy.matcher import Matcher

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

    # General inquiry if no specific keywords
    if any(ent.label_ == 'GPE' for ent in doc.ents) or re.search(r'\b\d+(\.\d+)?\s*BAM\b', message) or any(token.pos_ in ['NOUN', 'ADJ'] for token in doc):
        return 'SPECIFIC_INQUIRY'
    else:
        return 'GENERAL_INQUIRY'