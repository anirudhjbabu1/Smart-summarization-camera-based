import json
from collections import Counter
from datetime import datetime
import sys
import os

# --- Configuration ---
SUMMARY_OUTPUT_FILENAME = "ocr_narrative_summary.txt"
ACTIVITY_THRESHOLD = 50  # Words (non-duplicate count)
GAP_THRESHOLD_SECONDS = 5 # Time gap in seconds

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

def generate_narrative_summary(log_data):
    """Generates a structured, narrative summary from the log data."""
    if not log_data:
        return "No text detection events were recorded. Cannot generate summary."

    # --- 1. Preparation and Time Metrics ---
    
    # Parse all timestamps first
    timed_records = []
    for record in log_data:
        try:
            record['dt_time'] = parse_timestamp(record['timestamp'])
            timed_records.append(record)
        except ValueError:
            print(f"[WARNING] Skipping record due to bad timestamp: {record['timestamp']}")

    if not timed_records:
        return "No valid time data found in the log. Cannot generate summary."

    # Sort data by time
    timed_records.sort(key=lambda x: x['dt_time'])
    
    # FIX: Define and calculate total_words_read here
    total_words_read = sum(record['non_duplicate_count'] for record in timed_records)

    start_time = timed_records[0]['dt_time'].strftime('%H:%M:%S.%f')[:-3]
    end_time = timed_records[-1]['dt_time'].strftime('%H:%M:%S.%f')[:-3]
    total_duration = (timed_records[-1]['dt_time'] - timed_records[0]['dt_time']).total_seconds()
    
    # --- 2. Content Aggregation ---
    
    all_content_words = []
    for record in timed_records:
        all_content_words.extend(record['detected_words_list'])

    word_counts = Counter(all_content_words)
    top_content_words = [word for word, count in word_counts.most_common(10)]
    
    # Infer Content: Use the top 5 most frequent words to describe the content
    inferred_content_keywords = ', '.join(top_content_words[:5])
    
    # --- 3. Identifying Important Events (Activity/Gaps) ---
    important_events = []
    
    # Check for sustained high activity (many words detected in a single frame)
    for i, record in enumerate(timed_records):
        if record['non_duplicate_count'] >= ACTIVITY_THRESHOLD:
            important_events.append(f"- HIGH ACTIVITY at {record['timestamp']}: {record['non_duplicate_count']} words detected. (Possible large document/email/page)")

        # Check for significant gaps (indicates a pause or a switch between documents)
        if i > 0:
            time_diff = (record['dt_time'] - timed_records[i-1]['dt_time']).total_seconds()
            if time_diff >= GAP_THRESHOLD_SECONDS:
                important_events.append(f"- PAUSE/BREAK between {timed_records[i-1]['timestamp']} and {record['timestamp']} ({time_diff:.1f} seconds). (Possible switch of content)")

    if not important_events:
        important_events.append("- No significant pauses or high-activity events detected based on set thresholds.")

    # --- 4. Narrative Assembly ---

    narrative = "\n\n"
    narrative += "## Session Overview\n"
    narrative += "------------------------------------------------\n"
    
    # When did I start/stop?
    narrative += f"**[START TIME]** The OCR session began at **{start_time}**.\n"
    narrative += f"**[END TIME]** The last detection was recorded at **{end_time}**.\n"
    narrative += f"**[DURATION]** Total time span covered by detections: **{total_duration:.2f} seconds**.\n"
    narrative += "\n"
    
    # What was the content about?
    narrative += "## Inferred Content Summary\n"
    narrative += "------------------------------------------------\n"
    narrative += f"Based on the most frequently detected words, the content was likely related to:\n"
    narrative += f"**TOP CONTENT THEMES:** {inferred_content_keywords}\n\n"
    narrative += f"**TOP 10 WORDS READ (Overall):** {', '.join(top_content_words)}\n"
    narrative += f"**TOTAL UNIQUE WORDS:** {len(word_counts)}\n"
    
    # This line now works because total_words_read is defined
    narrative += f"**TOTAL WORDS READ (All Events):** {total_words_read}\n" 
    narrative += "\n"
    
    # Was there anything important?
    narrative += "## Important Reading Events (High Activity or Significant Pauses)\n"
    narrative += "------------------------------------------------\n"
    narrative += "The following events may indicate a large document/mail or a switch in reading material:\n"
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
        print("Usage: python3 ocr_narrative_analyzer.py <path/to/ocr_event_log_YYYYMMDD_HHMMSS.json>")
        sys.exit(1)
        
    log_filepath = sys.argv[1]
    
    # 1. Load Data
    data = load_log_file(log_filepath)
    
    # 2. Generate Narrative
    final_report = generate_narrative_summary(data)
    
    # 3. Save Report
    save_summary_file(final_report)
