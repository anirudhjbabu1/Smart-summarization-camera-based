import cv2
import numpy as np
import re
import time
import json 
from collections import Counter
import sys

# Use try-except for external libraries
try:
    import pytesseract
except ImportError:
    print("Error: pytesseract not found. Please install it using: pip3 install pytesseract")
    sys.exit(1)

# --- Configuration ---
CAMERA_INDEX = 0  # 0 is usually the default webcam. Change to 1 or 2 if needed.
LOG_DATA = []     # List to store ONLY detection events

# --- Functions ---
def count_and_extract_keywords(text):
    """
    Cleans text, removes consecutive duplicates, counts words, and extracts keywords.
    """
    # 1. Cleaning the text
    cleaned_text = re.sub(r'[^\w\s]', '', text).lower()
    all_detected_words = cleaned_text.split()
    all_detected_words = [word for word in all_detected_words if len(word) > 1]
    
    total_detected_count = len(all_detected_words)
    
    # 2. CONSECUTIVE Duplicate Deletion Logic
    words_no_consecutive_duplicates = []
    if all_detected_words:
        words_no_consecutive_duplicates.append(all_detected_words[0])
        for i in range(1, len(all_detected_words)):
            if all_detected_words[i] != all_detected_words[i-1]:
                words_no_consecutive_duplicates.append(all_detected_words[i])

    non_duplicate_count = len(words_no_consecutive_duplicates)
    
    # 3. Keyword Extraction (Frequency-based)
    stop_words = set(['the', 'a', 'an', 'is', 'it', 'to', 'and', 'or', 'of', 'in', 
                      'for', 'with', 'on', 'at', 'by', 'this', 'that', 'we', 'are', 
                      'be', 'will', 'was', 'as', 's', 't', 'd', 'm', 'i'])
    
    filtered_words = [word for word in words_no_consecutive_duplicates if word not in stop_words]
    word_freq = Counter(filtered_words)
    
    keywords = [item[0] for item in word_freq.most_common(5)]
    
    return total_detected_count, non_duplicate_count, keywords, words_no_consecutive_duplicates

def save_log_data(data):
    """Saves the accumulated list of detection events to a JSON file."""
    if not data:
        print("[INFO] No text detection events were recorded during this run.")
        return
        
    filename = f"ocr_event_log_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\n[SUCCESS] Saved {len(data)} detection events to {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save JSON file: {e}")


# --- Application Setup ---

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"FATAL ERROR: Could not open camera with index {CAMERA_INDEX}.")
    sys.exit(1)

print("--- Real-Time OCR Started (Event-Based Logging) ---")
print("Press 'q' to exit. Log will auto-save only frames where text was detected.")

try:
    # --- Main Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from camera. Breaking loop...")
            break

        # OCR Pre-processing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_frame = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)
        
        # Perform OCR
        extracted_text = pytesseract.image_to_string(binary_frame)
        
        # Analyze text
        total_count, non_duplicate_count, keywords, word_list = count_and_extract_keywords(extracted_text)

        # --- Event-Based Logging Logic ---
        # Only log the data if at least one word was detected after cleaning (i.e., non_duplicate_count > 0)
        if non_duplicate_count > 0:
            current_time_str = time.strftime('%H:%M:%S.%f')[:-3] # Time with milliseconds
            
            # Create a structured record for the current frame
            frame_record = {
                "timestamp": current_time_str,
                "total_words_detected": total_count,
                "non_duplicate_count": non_duplicate_count,
                "keywords": keywords,
                "detected_words_list": word_list # All detected words from this frame (consecutive duplicates removed)
            }
            LOG_DATA.append(frame_record)
            
            # Optional: Print a confirmation of the event
            # print(f"[{current_time_str}] Detected {non_duplicate_count} words.")


        # --- Display Results on the Frame ---
        display_color = (0, 255, 0) if non_duplicate_count > 0 else (0, 0, 255) # Green if detected, Red if not

        cv2.putText(frame, f"Words Detected: {non_duplicate_count}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
        cv2.putText(frame, f"Top Keywords: {', '.join(keywords)}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)
        cv2.putText(frame, f"TIME: {time.strftime('%H:%M:%S')}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to Quit (Auto-Saves Log)", (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Live Camera OCR', frame)

        # Check for 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"\n[FATAL RUNTIME ERROR] Program crashed: {e}")

finally:
    # --- AUTOMATIC SAVE AND CLEANUP ---
    print("\n--- Initiating Automatic Save and Cleanup ---")
    save_log_data(LOG_DATA)
    cap.release()
    cv2.destroyAllWindows()
    print("--- Application Closed Successfully ---")
