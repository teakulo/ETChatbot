import re
import numpy as np
import pandas as pd
import spacy
import csv
import dateparser
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from config import categorical_features

# -----------------------------
# Initializations
# -----------------------------
nlp = spacy.load("en_core_web_sm")
vectorizer = TfidfVectorizer(max_features=50)
one_hot = OneHotEncoder(handle_unknown='ignore')

# -----------------------------
# Function Definitions
# -----------------------------

def parse_date(date_string):
    return dateparser.parse(date_string)

def format_events_info(events):
    # Simplified to return just the names of the events
    return [event.get('name', 'N/A') for event in events]

def extract_message_entities(message):
    doc = nlp(message)

    # Extract locations (cities and venues)
    locations = [ent.text.lower() for ent in doc.ents if ent.label_ == 'GPE']

    # Extract and process dates
    extracted_dates = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
    dates = []
    for date in extracted_dates:
        parsed_date = dateparser.parse(date)
        if parsed_date:
            dates.append(parsed_date.strftime('%Y-%m-%d'))

    # Extract event keywords
    keywords = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]

    # Extract prices using a regular expression
    # This regex looks for numbers followed by 'BAM'
    price_pattern = r'\b\d+\s*BAM\b'
    prices = re.findall(price_pattern, message)

    # Combine all extracted information into a dictionary
    extracted_entities = {
        'locations': locations,
        'dates': dates,
        'keywords': keywords,
        'prices': prices
    }

    return extracted_entities

def load_events_data(csv_file_path):
    events_dataset = []
    try:
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                essential_keys = ['name', 'city', 'category', 'genre']
                if not all(key in row and row[key] and row[key].strip() for key in essential_keys):
                    print(f"Warning: Event missing essential information in CSV file at row: {row}")
                    continue
                for key in essential_keys:
                    row[key] = row[key].lower().strip()

                events_dataset.append(row)
    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found.")
    except Exception as e:
        print(f"Error loading data: {e}")
    return events_dataset


def extract_keywords_from_descriptions(descriptions, top_n=5):
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    return feature_array[tfidf_sorting][:top_n]

def extract_genre_keywords(events, top_n=5):
    descriptions = ' '.join([event['description'] for event in events]).lower().split()
    word_count = Counter(descriptions)
    stopwords = set(spacy.lang.en.stop_words.STOP_WORDS)
    filtered_words = [word for word in word_count if word not in stopwords and word.isalpha()]
    return sorted(filtered_words, key=lambda word: word_count[word], reverse=True)[:top_n]

def classify_query(query):
    if any(word in query.lower() for word in ['event', 'concert', 'show', 'exhibition']):
        return 'event_recommendation'
    elif any(word in query.lower() for word in ['help', 'how', 'what']):
        return 'help'
    else:
        return 'general'


def event_matches_criteria(event, extracted_entities):
    # Not all criteria need to match, but there should be at least one match
    matches_location = any(loc in [event.get('city', '').lower(), event.get('venue', '').lower()] for loc in
                           extracted_entities['locations'])
    matches_date = any(date in event.get('start_time', '') for date in extracted_entities['dates'])
    matches_keywords = any(
        keyword in event.get('description', '').lower() for keyword in extracted_entities['keywords'])

    return matches_location or matches_date or matches_keywords


def encode_query(query, event_keywords=None, genre_keywords=None):
    doc = nlp(query.lower())
    query_type = classify_query(query)
    if query_type != 'event_recommendation':
        return query_type, None

    encoded_features = []
    locations = [ent.text.lower() for ent in doc.ents if ent.label_ == 'GPE']
    categories = [keyword for keyword in event_keywords if keyword in query.lower()]
    genres = [genre for genre in genre_keywords if genre in query.lower()]

    query_data = {
        'city': [locations[0] if locations else 'unknown'],
        'category': [categories[0] if categories else 'unknown'],
        'genre': [genres[0] if genres else 'unknown']
    }

    query_df = pd.DataFrame(query_data)
    for feature in categorical_features:
        if feature not in query_df.columns:
            query_df[feature] = ['unknown']

    encoded_cat_features = one_hot.transform(query_df).toarray()
    encoded_features.extend(encoded_cat_features.flatten())

    encoded_text_features = vectorizer.transform([query]).toarray()
    encoded_features.extend(encoded_text_features.flatten())

    return np.array(encoded_features)

def recommend_events_knn(encoded_query):
    distances, indices = knn.kneighbors([encoded_query])
    recommended_event_indices = indices[0]
    print(f"kNN Distances: {distances}")
    print(f"kNN Indices: {indices}")
    return [df.iloc[i]['name'] for i in recommended_event_indices]

# Data Preprocessing
events_dataset = load_events_data('sample_events.csv')
df = pd.DataFrame(events_dataset)
df[categorical_features] = df[categorical_features].apply(lambda x: x.fillna('unknown'))
one_hot.fit(df[categorical_features])
features = one_hot.transform(df[categorical_features]).toarray()
tfidf_description = vectorizer.fit_transform(df['description'])
features = np.hstack((features, tfidf_description.toarray()))

knn = NearestNeighbors(n_neighbors=5)
knn.fit(features)