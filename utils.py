import logging
import re
import spacy
import csv
import dateparser
from dateparser.search import search_dates
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import config

nlp = spacy.load("en_core_web_sm")

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
    config.events_dataset = []
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

                config.events_dataset.append(row)

    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found.")
    except Exception as e:
        print(f"Error loading data: {e}")

    return config.events_dataset


def get_numeric_price(price_str):
    # Extracts the numeric part from a price string and appends "BAM"
    match = re.search(r'\b\d+(\.\d+)?', price_str)
    return f"{float(match.group())} BAM" if match else "N/A"


def event_matches_criteria(event, extracted_entities):
    logging.debug(f"Processing event: {event}")

    # Explicitly define keywords to be completely ignored in the match process
    exclude_keywords = {'event'}

    # Extract keywords, ensuring excluded ones are not considered even for matching attempt
    keywords_to_match = set(extracted_entities.get('keywords', [])) - exclude_keywords

    # Compile event information into a searchable string
    event_info = ' '.join(
        [event.get(field, '').lower() for field in ['description', 'city', 'venue', 'category', 'name']]
    )

    # Check if each keyword to match is found in the event information
    matched_keywords = {keyword for keyword in keywords_to_match if keyword.lower() in event_info}

    # A successful match occurs when all keywords to match are found within the event information
    matches_keywords = matched_keywords == keywords_to_match

    logging.debug(f"Attempting to Match Keywords: {keywords_to_match}, Successfully Matched: {matched_keywords}, Matches Keywords: {matches_keywords}")

    # Assuming time frame and prices match by default unless specifics are provided
    matches_time_frame = True
    matches_price = True

    # Add your specific logic for matching time frame and prices if necessary...

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


