import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import csv
import dateparser
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from config import categorical_features


def parse_date(date_string):
    return dateparser.parse(date_string)

nlp = spacy.load("en_core_web_sm")
vectorizer = TfidfVectorizer(max_features=50)
one_hot = OneHotEncoder(handle_unknown='ignore')
knn = NearestNeighbors(n_neighbors=5)

def format_events_info(events):
    if not events:
        return "No events available."

    formatted_info = "Upcoming Events:\n"
    formatted_info += "-" * 20 + "\n"

    for event in events:
        formatted_info += f"Name: {event.get('name', 'N/A')}\n"
        formatted_info += f"City: {event.get('city', 'N/A')}\n"
        formatted_info += f"Genre: {event.get('genre', 'N/A')}\n"
        formatted_info += f"Price: {event.get('price', 'N/A')}\n"
        formatted_info += "-" * 20 + "\n"

    return formatted_info
def handle_event_recommendation(user_message, events_dataset):
    parsed_date = parse_date(user_message)
    if parsed_date:
        formatted_date = parsed_date.strftime('%Y-%m-%d')
        recommended_events = [event for event in events_dataset if event['start_time'].startswith(formatted_date)]
        if recommended_events:
            events_info = format_events_info(recommended_events)
            return events_info
        else:
            return "No events found around that date."
    else:
        event_keywords = extract_keywords_from_descriptions([event['description'] for event in events_dataset])
        genre_keywords = extract_genre_keywords(events_dataset)
        encoded_query = encode_query(user_message, event_keywords, genre_keywords)
        return recommend_events_knn(encoded_query, events_dataset)

def extract_entities(message):
    doc = nlp(message)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return entities
def load_events_data(csv_file_path):
    events_dataset = []
    try:
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Convert categorical data to lowercase
                row['city'] = row['city'].lower() if row['city'] else ''
                row['category'] = row['category'].lower() if row['category'] else ''
                row['genre'] = row['genre'].lower() if row['genre'] else ''
                events_dataset.append(row)

    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found.")
    except Exception as e:
        print(f"Error loading data: {e}")
    return events_dataset


def extract_keywords_from_descriptions(descriptions, top_n=10):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    top_keywords = feature_array[tfidf_sorting][:top_n]
    return top_keywords

def extract_genre_keywords(events, top_n=10):
    descriptions = ' '.join([event['description'] for event in events]).lower().split()
    word_count = Counter(descriptions)
    stopwords = set(spacy.lang.en.stop_words.STOP_WORDS)
    filtered_words = [word for word in word_count if word not in stopwords and word.isalpha()]
    genre_keywords = sorted(filtered_words, key=lambda word: word_count[word], reverse=True)[:top_n]
    return genre_keywords

def classify_query(query):
    # Simple keyword-based classification
    if any(word in query.lower() for word in ['event', 'concert', 'show', 'exhibition']):
        return 'event_recommendation'
    elif any(word in query.lower() for word in ['help', 'how', 'what']):
        return 'help'
    else:
        return 'general'


def encode_query(query, event_keywords, genre_keywords, one_hot, vectorizer, categorical_features):
    # Process the query with spaCy
    doc = nlp(query.lower())
    query_type = classify_query(query)

    # If the query is not of type 'event_recommendation', return None for encoded features
    if query_type != 'event_recommendation':
        return query_type, None

    # Initialize an empty list to hold the encoded features
    encoded_features = []

    # Extract location, category, and genre from the query
    locations = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    categories = [keyword for keyword in event_keywords if keyword in query.lower()]
    genres = [genre for genre in genre_keywords if genre in query.lower()]

    # Prepare data for one-hot encoding, using 'unknown' if no data is present
    query_data = {
        'city': [locations[0] if locations else 'unknown'],
        'category': [categories[0] if categories else 'unknown'],
        'genre': [genres[0] if genres else 'unknown']
    }

    # Create DataFrame from query_data for one-hot encoding
    query_df = pd.DataFrame(query_data)

    # Ensure the DataFrame columns match the trained OneHotEncoder feature names
    for feature in categorical_features:
        if feature not in query_df.columns:
            query_df[feature] = ['unknown']  # Add missing feature as 'unknown'

    # One-hot encode categorical features
    encoded_cat_features = one_hot.transform(query_df).toarray()
    encoded_features.extend(encoded_cat_features.flatten())

    # TF-IDF encode the entire query as a textual feature
    encoded_text_features = vectorizer.transform([query]).toarray()
    encoded_features.extend(encoded_text_features.flatten())

    # Return the encoded features as a numpy array
    return np.array(encoded_features)

# Set up the features for kNN after loading data
events_dataset = load_events_data('sample_events.csv')
df = pd.DataFrame(events_dataset)
df[categorical_features] = df[categorical_features].apply(lambda x: x.fillna('unknown'))
one_hot.fit(df[categorical_features])
features = one_hot.transform(df[categorical_features]).toarray()
tfidf_description = vectorizer.fit_transform(df['description'])
features = np.hstack((features, tfidf_description.toarray()))
knn.fit(features)