import json
from collections import Counter
from datetime import datetime
import sys
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from heapq import nlargest # Used for finding the N largest elements (sentences)

# --- Configuration ---
SUMMARY_OUTPUT_FILENAME = "ocr_smart_narrative_summary.txt"
ACTIVITY_THRESHOLD = 50  # Words
GAP_THRESHOLD_SECONDS = 5 # Time gap in seconds

# --- Helper Functions (Loading, Parsing) ---

def load_log_file(filename):
    """Loads the JSON log file and handles file errors."""
    if not os.path.exists(filename):
        print(f"Error: Log file not found at '{filename}'.")
        print("Please ensure the file exists and the path is correct.")
        sys.exit(1)
        
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"[INFO] Successfully loaded {len(data)} detection events.")
        return data
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{filename}'. Is the file corrupted?")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading the file: {e}")
        sys.exit(1)

def parse_timestamp(time_str):
    """Safely attempts to parse timestamps with and without milliseconds."""
    try:
        return datetime.strptime(time_str, '%H:%M:%S.%f')
    except ValueError:
        try:
            return datetime.strptime(time_str, '%H:%M:%S')
        except ValueError as e:
            raise ValueError(f"Could not parse time data '{time_str}'.") from e

# --- New NLP Core Function ---

def generate_extractive_summary(all_words, num_sentences=4):
    """
    Creates an extractive summary by calculating the frequency of words 
    and scoring sentences based on word frequency.
    
    Since the OCR output provides words list, we reconstruct the text as a single string.
    """
    if not all_words:
        return "Not enough coherent text detected to generate a narrative summary."

    # 1. Reconstruct text from the clean word list
    text = " ".join(all_words)
    
    # 2. Tokenize and Pre-process
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # Check if we have enough sentences to summarize
    if len(sentences) < num_sentences:
        return " ".join(sentences) # Return all text if too short

    # 3. Calculate Word Frequencies
    stop_words = set(stopwords.words('english'))
    word_frequencies = {}
    
    for word in words:
        if word.isalnum() and word not in stop_words:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
                
    # Max frequency for normalization
    max_frequency = max(word_frequencies.values()) if word_frequencies else 1
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / max_frequency)
        
    # 4. Score Sentences
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies.keys():
                if len(sentence.split(' ')) < 30: # Avoid overly long or fragmented OCR sentences
                    if sentence not in sentence_scores.keys():
                        sentence_scores[sentence] = word_frequencies[word]
                    else:
                        sentence_scores[sentence] += word_frequencies[word]

    # 5. Extract Top N Sentences
    # nlargest returns the sentences with the highest scores
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    return " ".join(summary_sentences)

# --- Main Analysis Function ---

def generate_narrative_summary(log_data):
    """Generates a structured, narrative summary from the log data."""
    if not log_data:
        return "No text detection events were recorded. Cannot generate summary."

    timed_records = []
    for record in log_data:
        try:
            record['dt_time'] = parse_timestamp(record['timestamp'])
            timed_records.append(record)
        except ValueError:
            pass # Skip bad timestamps

    if not timed_records:
        return "No valid time data found in the log. Cannot generate summary."

    timed_records.sort(key=lambda x: x['dt_time'])

    # --- Time Metrics ---
    start_time = timed_records[0]['dt_time'].strftime('%H:%M:%S.%f')[:-3]
    end_time = timed_records[-1]['dt_time'].strftime('%H:%M:%S.%f')[:-3]
    total_duration = (timed_records[-1]['dt_time'] - timed_records[0]['dt_time']).total_seconds()
    
    # --- Content Aggregation ---
    all_content_words = []
    for record in timed_records:
        all_content_words.extend(record['detected_words_list'])

    word_counts = Counter(all_content_words)
    total_words_read = sum(record['non_duplicate_count'] for record in timed_records)
    top_content_words = [word for word, count in word_counts.most_common(10)]
    
    # --- NLP Summarization ---
    content_summary_narrative = generate_extractive_summary(all_content_words)
    
    # --- Identifying Important Events ---
    important_events = []
    
    for i, record in enumerate(timed_records):
        if record['non_duplicate_count'] >= ACTIVITY_THRESHOLD:
            important_events.append(f"- HIGH ACTIVITY at {record['timestamp']}: {record['non_duplicate_count']} words detected. (Possible dense document/mail)")

        if i > 0:
            time_diff = (record['dt_time'] - timed_records[i-1]['dt_time']).total_seconds()
            if time_diff >= GAP_THRESHOLD_SECONDS:
                important_events.append(f"- PAUSE/BREAK between {timed_records[i-1]['timestamp']} and {record['timestamp']} ({time_diff:.1f} seconds). (Possible switch of content)")

    if not important_events:
        important_events.append("- No significant pauses or high-activity events detected based on set thresholds.")

    # --- Narrative Assembly ---

    narrative = f"OCR Smart Analysis Report (Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n"
    narrative += "=" * 60 + "\n"
    
    narrative += "## 1. Content Summary (What was read?)\n"
    narrative += "------------------------------------------------\n"
    narrative += "This section provides a summary of the content by identifying the most important sentences:\n\n"
    narrative += f"**NARRATIVE:** {content_summary_narrative}\n\n"
    narrative += f"**TOP 10 CONTENT WORDS:** {', '.join(top_content_words)}\n"
    
    narrative += "\n## 2. Session Metrics (When did I read?)\n"
    narrative += "------------------------------------------------\n"
    narrative += f"**[START TIME]** The OCR session began at **{start_time}**.\n"
    narrative += f"**[END TIME]** The last detection was recorded at **{end_time}**.\n"
    narrative += f"**[TOTAL DURATION]** Total time span covered by detections: **{total_duration:.2f} seconds**.\n"
    narrative += f"**[TOTAL WORDS READ]** Total words processed (non-consecutive duplicates): **{total_words_read}**.\n"
    
    narrative += "\n## 3. Key Events (Was there anything important?)\n"
    narrative += "------------------------------------------------\n"
    narrative += "The following events suggest a large document (high word count) or a major pause:\n"
    narrative += "\n".join(important_events)
    narrative += "\n\n"
    
    return narrative


def save_summary_file(report):
    """Saves the final report to a text file."""
    try:
        with open(SUMMARY_OUTPUT_FILENAME, 'w') as f:
            f.write(report)
        print(f"\n[SUCCESS] Narrative analysis report saved to '{SUMMARY_OUTPUT_FILENAME}'")
    except Exception as e:
        print(f"Error: Could not save the summary file. {e}")

# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ocr_smart_analyzer.py <path/to/ocr_event_log_YYYYMMDD_HHMMSS.json>")
        sys.exit(1)
        
    log_filepath = sys.argv[1]
    
    # 1. Load Data
    data = load_log_file(log_filepath)
    
    # 2. Generate Narrative
    final_report = generate_narrative_summary(data)
    
    # 3. Save Report
    save_summary_file(final_report)
