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

def extract_entities(message):
    doc = nlp(message)
    return [(ent.text, ent.label_) for ent in doc.ents]


def load_events_data(csv_file_path):
    events_dataset = []
    try:
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Ensure that the row is a dictionary and the key exists before trying to modify it
                if 'city' in row and row['city']:
                    row['city'] = row['city'].lower()
                else:
                    row['city'] = ''

                if 'category' in row and row['category']:
                    row['category'] = row['category'].lower()
                else:
                    row['category'] = ''

                if 'genre' in row and row['genre']:
                    row['genre'] = row['genre'].lower()
                else:
                    row['genre'] = ''

                events_dataset.append(row)
    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found.")
    except Exception as e:
        print(f"Error loading data: {e}")
    return events_dataset

def extract_keywords_from_descriptions(descriptions, top_n=5):
    vectorizer = TfidfVectorizer()
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

def encode_query(query, event_keywords=None, genre_keywords=None):
    doc = nlp(query.lower())
    query_type = classify_query(query)

    if query_type != 'event_recommendation':
        return query_type, None

    encoded_features = []

    locations = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
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
    return [df.iloc[i]['name'] for i in indices[0]]

# -----------------------------
# Data Preprocessing
# -----------------------------

events_dataset = load_events_data('sample_events.csv')
df = pd.DataFrame(events_dataset)
df[categorical_features] = df[categorical_features].apply(lambda x: x.fillna('unknown'))
one_hot.fit(df[categorical_features])
features = one_hot.transform(df[categorical_features]).toarray()
tfidf_description = vectorizer.fit_transform(df['description'])
features = np.hstack((features, tfidf_description.toarray()))

knn = NearestNeighbors(n_neighbors=5)
knn.fit(features)
