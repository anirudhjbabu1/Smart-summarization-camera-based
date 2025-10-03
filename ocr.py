import cv2
import pytesseract
import numpy as np
import re
from collections import Counter

# --- Configuration ---
# Set the path to the Tesseract executable (This is usually not needed on Ubuntu 
# if installed via apt-get, but good for explicit paths if necessary)
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' 

# --- Camera Setup ---
# Use 0 for the default camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# --- Functions ---
def count_and_extract_keywords(text):
    # 1. Cleaning the text: Remove punctuation and convert to lowercase
    cleaned_text = re.sub(r'[^\w\s]', '', text).lower()
    
    # 2. Split into words
    words = cleaned_text.split()
    
    # 3. Word Count
    word_count = len(words)
    
    # 4. Keyword Extraction (Simple Frequency-based method, ignoring common words)
    stop_words = set(['the', 'a', 'an', 'is', 'it', 'to', 'and', 'or', 'of', 'in', 'for', 'with', 'on'])
    
    # Filter out stop words and single-letter words
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    
    # Count frequency
    word_freq = Counter(filtered_words)
    
    # Get the top N keywords (e.g., top 5)
    keywords = [item[0] for item in word_freq.most_common(5)]
    
    return word_count, keywords

# --- Main Loop ---
print("--- Real-Time OCR Started ---")
print("Press 'q' to exit the application.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is not read correctly, break
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the image to grayscale for better OCR results
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a threshold to get a binary image (optional, but often improves OCR)
    _, binary_frame = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)

    # Perform OCR
    extracted_text = pytesseract.image_to_string(binary_frame)
    
    # Count words and extract keywords
    word_count, keywords = count_and_extract_keywords(extracted_text)

    # --- Display Results on the Frame ---
    # Display the word count
    cv2.putText(frame, f"Words: {word_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the keywords
    cv2.putText(frame, f"Keywords: {', '.join(keywords)}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Live Camera OCR', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("--- Application Closed ---")
