import calendar
import logging
import re
import numpy as np
import pandas as pd
import spacy
import csv
import dateparser
from collections import Counter
from dateparser.search import search_dates
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from config import categorical_features
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


# -----------------------------
# Initializations
# -----------------------------
nlp = spacy.load("en_core_web_sm")
vectorizer = TfidfVectorizer(max_features=50)
one_hot = OneHotEncoder(handle_unknown='ignore')

# -----------------------------
# Function Definitions
# -----------------------------

def extract_message_entities(message):
    doc = nlp(message)
    logging.debug(f"Analyzing message: {message}")

    keywords = [ent.text.lower() for ent in doc.ents if ent.label_ == 'GPE']
    general_keywords = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    keywords.extend(general_keywords)
    logging.debug(f"Extracted keywords: {keywords}")

    # Use the custom get_time_frame function to interpret time-related phrases
    time_frame = get_time_frame(message)
    logging.debug(f"Extracted time frame: {time_frame}")


    price_pattern = r'\b\d+(\.\d+)?\s*BAM\b'
    prices = re.findall(price_pattern, message)
    logging.debug(f"Extracted prices: {prices}")

    return {'keywords': keywords, 'time_frame': time_frame, 'prices': prices}

def load_events_data(csv_file_path):
    events_dataset = []
    try:
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                essential_keys = ['name', 'start_time', 'end_time', 'venue', 'city', 'category', 'genre']
                if not all(key in row and row[key].strip() for key in essential_keys):
                    print(f"Warning: Event missing essential information in CSV file at row: {row}")
                    continue

                for key in essential_keys:
                    row[key] = row[key].strip()

                # Directly assign the last column value to price
                row['price'] = row.get('price', 'N/A')

                events_dataset.append(row)

    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found.")
    except Exception as e:
        print(f"Error loading data: {e}")

    return events_dataset




def get_numeric_price(price_str):
    # Extracts the numeric part from a price string and appends "BAM"
    match = re.search(r'\b\d+(\.\d+)?', price_str)
    return f"{float(match.group())} BAM" if match else "N/A"

def event_matches_criteria(event, extracted_entities):
    logging.debug(f"Processing event: {event}")

    # Check for keyword match (e.g., 'concert', 'Sarajevo')
    event_info = ' '.join([event.get(field, '').lower() for field in ['description', 'city', 'venue', 'category']])
    keywords = extracted_entities.get('keywords', [])
    matches_keywords = all(keyword in event_info for keyword in keywords) if keywords else True

    # Check for time frame match
    event_start = dateparser.parse(event.get('start_time'))
    time_frame = extracted_entities.get('time_frame')
    matches_time_frame = True
    if time_frame and event_start:
        start_frame, end_frame = time_frame
        matches_time_frame = start_frame <= event_start <= end_frame
    logging.debug(f"Matches time frame: {matches_time_frame} (Event Date: {event_start}, Time Frame: {time_frame})")

    # Check for price match
    event_price = get_numeric_price(event.get('price', ''))
    extracted_prices = [get_numeric_price(price) for price in extracted_entities.get('prices', [])]
    matches_price = True if not extracted_prices else any(price == event_price for price in extracted_prices if price is not None)
    logging.debug(f"Matches price: {matches_price}")


    return matches_keywords and matches_time_frame and matches_price

def format_events_info(events):
    formatted_events = "<table style='width:100%; border-collapse: collapse;'>"
    formatted_events += "<tr style='background-color: #f2f2f2;'><th>Name</th><th>Date</th><th>Venue</th><th>City</th><th>Category</th></tr>"

    for event in events:
        start_date = dateparser.parse(event.get('start_time')).strftime('%d-%m-%Y') if event.get('start_time') else 'Unknown date'

        formatted_events += "<tr style='border-bottom: 1px solid #ddd;'>"
        formatted_events += f"<td>{event.get('name', 'Unnamed Event')}</td>"
        formatted_events += f"<td>{start_date}</td>"
        formatted_events += f"<td>{event.get('venue', 'Unknown venue')}</td>"
        formatted_events += f"<td>{event.get('city', 'Unknown city')}</td>"
        formatted_events += f"<td>{event.get('category', 'N/A')}</td>"
        formatted_events += "</tr>"

    formatted_events += "</table>"
    return formatted_events


def get_time_frame(message):
    logging.debug(f"Received message for time frame extraction: {message}")
    current_date = datetime.now()

    # Regular expressions for different time phrases
    next_year_pattern = r'next year'
    next_month_pattern = r'next month'
    next_week_pattern = r'next week'
    in_days_pattern = r'in (\d+) days'
    in_weeks_pattern = r'in (\d+) weeks'
    in_months_pattern = r'in (\d+) months'

    if re.search(next_year_pattern, message.lower()):
        start_date = datetime(current_date.year + 1, 1, 1)
        end_date = datetime(current_date.year + 1, 12, 31)
        return (start_date, end_date)

    if re.search(next_month_pattern, message.lower()):
        start_date = current_date + relativedelta(months=+1)
        end_date = start_date + relativedelta(months=+1, days=-1)
        return (start_date, end_date)

    if re.search(next_week_pattern, message.lower()):
        start_date = current_date + timedelta(weeks=1)
        end_date = start_date + timedelta(days=6)
        return (start_date, end_date)

    days_match = re.search(in_days_pattern, message.lower())
    weeks_match = re.search(in_weeks_pattern, message.lower())
    months_match = re.search(in_months_pattern, message.lower())

    if days_match:
        days_ahead = int(days_match.group(1))
        target_date = current_date + timedelta(days=days_ahead)
        return (target_date, target_date)

    if weeks_match:
        weeks_ahead = int(weeks_match.group(1))
        target_date = current_date + timedelta(weeks=weeks_ahead)
        return (target_date, target_date)

    if months_match:
        months_ahead = int(months_match.group(1))
        target_date = current_date + relativedelta(months=+months_ahead)
        return (target_date, target_date)

    # Default date parsing using dateparser
    dates_found = search_dates(message, settings={'PREFER_DATES_FROM': 'future', 'DATE_ORDER': 'DMY'})
    if dates_found:
        return (dates_found[0][1], dates_found[0][1])

    return None





# def classify_query(query):
#     if any(word in query.lower() for word in ['event']):
#         return 'event_recommendation'
#     elif any(word in query.lower() for word in ['help', 'how', 'what']):
#         return 'help'
#     else:
#         return 'general'

# # Data Preprocessing
# events_dataset = load_events_data('sample_events.csv')
# df = pd.DataFrame(events_dataset)
# df[categorical_features] = df[categorical_features].apply(lambda x: x.fillna('unknown'))
# one_hot.fit(df[categorical_features])
# features = one_hot.transform(df[categorical_features]).toarray()
# tfidf_description = vectorizer.fit_transform(df['description'])
# features = np.hstack((features, tfidf_description.toarray()))

# def encode_query(query, event_keywords=None, genre_keywords=None):
#     doc = nlp(query.lower())
#     query_type = classify_query(query)
#     if query_type != 'event_recommendation':
#         return query_type, None
#
#     encoded_features = []
#     locations = [ent.text.lower() for ent in doc.ents if ent.label_ == 'GPE']
#     categories = [keyword for keyword in event_keywords if keyword in query.lower()]
#     genres = [genre for genre in genre_keywords if genre in query.lower()]
#
#     query_data = {
#         'city': [locations[0] if locations else 'unknown'],
#         'category': [categories[0] if categories else 'unknown'],
#         'genre': [genres[0] if genres else 'unknown']
#     }
#
#     query_df = pd.DataFrame(query_data)
#     for feature in categorical_features:
#         if feature not in query_df.columns:
#             query_df[feature] = ['unknown']
#
#     encoded_cat_features = one_hot.transform(query_df).toarray()
#     encoded_features.extend(encoded_cat_features.flatten())
#
#     encoded_text_features = vectorizer.transform([query]).toarray()
#     encoded_features.extend(encoded_text_features.flatten())
#
#     return np.array(encoded_features)
#


#def extract_keywords_from_descriptions(descriptions, top_n=5):
#     tfidf_matrix = vectorizer.fit_transform(descriptions)
#     feature_array = np.array(vectorizer.get_feature_names_out())
#     tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
#     return feature_array[tfidf_sorting][:top_n]
# def recommend_events_knn(encoded_query):
#     distances, indices = knn.kneighbors([encoded_query])
#     recommended_event_indices = indices[0]
#     print(f"kNN Distances: {distances}")
#     print(f"kNN Indices: {indices}")
#     return [df.iloc[i]['name'] for i in recommended_event_indices]
# def extract_genre_keywords(events, top_n=5):
#     descriptions = ' '.join([event['description'] for event in events]).lower().split()
#     word_count = Counter(descriptions)
#     stopwords = set(spacy.lang.en.stop_words.STOP_WORDS)
#     filtered_words = [word for word in word_count if word not in stopwords and word.isalpha()]
#     return sorted(filtered_words, key=lambda word: word_count[word], reverse=True)[:top_n]



#knn = NearestNeighbors(n_neighbors=5)
#knn.fit(features)