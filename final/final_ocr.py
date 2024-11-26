import pandas as pd
import re
import difflib
import pytesseract
from pytesseract import Output
import cv2
from typing import List, Tuple, Dict

class BoundingBox:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, text: str):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.text = text
        self.value = None
        self.unit = None
        self.infer_text()
        print(self.value)
    
    def get_centroid(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def infer_text(self):
        value, unit = self.text.split()
        self.value = float(value)
        self.unit = str(unit)

# Abbreviation map for units
abbreviation_map = {
    'in': 'inch',
    'cm': 'centimetre',
    'ft': 'foot',
    'm': 'metre',
    'mm': 'millimetre',
    'yd': 'yard',
    'g': 'gram',
    'kg': 'kilogram',
    'mg': 'milligram',
    'ug': 'microgram',
    'oz': 'ounce',
    'lb': 'pound',
    't': 'ton',
    'v': 'volt',
    'mv': 'millivolt',
    'kv': 'kilovolt',
    'w': 'watt',
    'kw': 'kilowatt',
    'ml': 'millilitre',
    'l': 'litre',
    'dl': 'decilitre',
    'cl': 'centilitre',
    'gal': 'gallon',
    'pt': 'pint',
    'qt': 'quart',
    'cu in': 'cubic inch',
    'cu ft': 'cubic foot',
}

entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'maximum_weight_recommendation': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre',
        'cubic foot',
        'cubic inch',
        'cup',
        'decilitre',
        'fluid ounce',
        'gallon',
        'imperial gallon',
        'litre',
        'microlitre',
        'millilitre',
        'pint',
        'quart'}
}

# Combine both abbreviations and full names into a single set for matching
valid_units = set(abbreviation_map.keys()).union(set(abbreviation_map.values()))

def correct_ocr_errors(text) -> str:
    # print("Input text:", text)
    # Common OCR errors and their corrections for specified units
    corrections = {
        # Centimetre
        "centimetre": "centimetre", "centimeter": "centimetre",
        "centimetres": "centimetre", "centimeters": "centimetre",
        "centlmetre": "centimetre", "centlmetres": "centimetre",
        "cm": "centimetre", "CM": "centimetre",
        # Foot
        "ft": "foot", "feet": "foot", "foots": "foot",
        # Millimetre
        "millimetre": "millimetre", "millimeter": "millimetre",
        "millimetres": "millimetre", "millimeters": "millimetre",
        "milimetre": "millimetre", "milimetres": "millimetre",
        "mm": "millimetre",
        # Metre
        "metre": "metre", "meter": "metre",
        "metres": "metre", "meters": "metre",
        "m": "metre",
        # Inch
        "inch": "inch", "inches": "inch",
        "inche": "inch", "inchs": "inch",
        "inoh": "inch", "lnch": "inch",
        "\"": "inch", 'Inchy': "inch", 
        "inchy": "inch",
        # Yard
        "yard": "yard", "yards": "yard",
        "yrd": "yard", "yrds": "yard",
        "yd": "yard", "yds": "yard",
    }

    # Function to correct numbers
    def correct_number(match):
        num = match.group(0)
        corrected = num.replace('O', '0').replace('l', '1').replace('I', '1')
        return corrected

    # Correct numbers (replace 'O' with '0', 'l' or 'I' with '1')
    text = re.sub(r'\d+', correct_number, text)

    # Correct spacing issues (e.g., "34. 5" to "34.5")
    text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)

    # Split the text into words
    words = text.split()

    # Process each word
    for i, word in enumerate(words):
        # Check if the word has a number followed by a unit
        match = re.match(r'(\d+)([a-zA-Z]+)', word)
        if match:
            number, unit = match.groups()
            lower_unit = unit.lower()
            
            # Check if the unit is in our corrections dictionary
            if lower_unit in corrections:
                corrected_unit = corrections[lower_unit]
                words[i] = f"{number} {corrected_unit}"
            else:
                # Use difflib to find the closest match for the unit
                close_matches = difflib.get_close_matches(lower_unit, corrections.keys(), n=1, cutoff=0.8)
                if close_matches:
                    corrected_unit = corrections[close_matches[0]]
                    words[i] = f"{number} {corrected_unit}"
        else:
            # Process standalone words without numbers
            lower_word = word.lower()
            if lower_word in corrections:
                words[i] = corrections[lower_word]
            else:
                close_matches = difflib.get_close_matches(lower_word, corrections.keys(), n=1, cutoff=0.8)
                if close_matches:
                    words[i] = corrections[close_matches[0]]

    # Join the words back into a string
    corrected_text = ' '.join(words)
    # print("Output text:", corrected_text)
    return corrected_text

def extract_value_from_ocr(ocr_text):
    if pd.isna(ocr_text) or not isinstance(ocr_text, str):
        return None  # Return None if OCR text is invalid

    # Pattern to capture floats/integers followed by units
    pattern = r'(\d+\.?\d*)\s*([a-zA-Z\s]+)'  
    matches = re.findall(pattern, ocr_text)

    # print(matches)

    extracted_value = None
    for value, unit in matches:
        value = str(float(value))
        unit = unit.strip().lower()  # Normalize the unit to lowercase
        
        # If the unit is an abbreviation, replace it with the full unit
        if unit in abbreviation_map:
            full_unit = abbreviation_map[unit]
        else:
            # If the unit is already a full unit, use it directly
            full_unit = unit if unit in abbreviation_map.values() else correct_ocr_errors(value + ' ' + unit)

        if full_unit:
            extracted_value = value + ' ' + full_unit
            break
    
    if not extracted_value or len(extracted_value.split()) != 2:
        return None

    # print("Extracted value:", extracted_value, ocr_text)
    # return extracted_value
    final_value, final_unit = extracted_value.split()
    if final_unit in entity_unit_map['width'] or final_unit in entity_unit_map['depth'] or final_unit in entity_unit_map['height']:
        return extracted_value

    return None

def extract_bboxes(img):
    # Load the image
    # img = cv2.imread(image_path)
    
    # Use pytesseract to get bounding boxes
    boxes = pytesseract.image_to_data(img, output_type=Output.DICT, config='--psm 12', lang='eng')
    
    # Extract bounding boxes coordinates and text
    n_boxes = len(boxes['text'])
    bboxes = []
    img = np.array(img)
    img_height, img_width = img.shape[:2]  # Get image dimensions for boundary checks
    
    for i in range(n_boxes):
        if boxes['text'][i].strip():
            # Original bounding box coordinates
            x, y, w, h = boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i]
            
            # Expand the bounding box by 25% in all directions
            x_expansion = int(0.25 * w)
            y_expansion = int(0.25 * h)
            
            # Calculate new coordinates
            x_new = max(0, x - x_expansion)  # Ensure not going out of image bounds
            y_new = max(0, y - y_expansion)
            w_new = min(img_width, x + w + x_expansion)  # Ensure not exceeding image width
            h_new = min(img_height, y + h + y_expansion)  # Ensure not exceeding image height
            
            bboxes.append([x_new, y_new, w_new, h_new])
    
    return img, bboxes


import numpy as np

def merge_bboxes(bboxes, max_bboxes=3):
    def compute_iou(box1, box2):
        # Compute Intersection Over Union (IoU) between two boxes
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def combine_bboxes(box1, box2):
        # Combine two boxes by taking the outermost coordinates
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])
        return [x1, y1, x2, y2]
    
    while len(bboxes) > max_bboxes:
        merged = False
        # Step 1: Always merge intersecting boxes first
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if compute_iou(bboxes[i], bboxes[j]) > 0.0:  # Intersecting boxes
                    new_box = combine_bboxes(bboxes[i], bboxes[j])
                    bboxes[i] = new_box
                    bboxes.pop(j)
                    merged = True
                    break
            if merged:
                break

        # Step 2: If no boxes were merged, use proximity to merge the closest pair
        if not merged:
            a = np.array(bboxes)
            centroids = np.array([(0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])) for box in bboxes])
            pairwise_distances = np.sqrt(np.sum(np.square(centroids[:, None] - centroids[None, :]), axis=-1))
            
            # Set diagonal to infinity to avoid merging a box with itself
            np.fill_diagonal(pairwise_distances, np.inf)
            
            # Find the closest pair of boxes and merge them
            i, j = np.unravel_index(np.argmin(pairwise_distances), pairwise_distances.shape)
            new_box = combine_bboxes(bboxes[i], bboxes[j])
            bboxes[i] = new_box
            bboxes.pop(j)

    return bboxes

import matplotlib.pyplot as plt
import cv2

def draw_bboxes(img, bboxes):
    # Convert BGR image (OpenCV) to RGB for Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw the final bounding boxes on the image
    for box in bboxes:
        cv2.rectangle(img_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    
    # Display the image using Matplotlib
    plt.figure(figsize=(3, 3))
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axes for better viewing
    plt.show()

def extract_text_from_bboxes(img, bboxes):
    text_bbox = []
    for box in bboxes:
        # Crop the region of the image inside the bounding box
        cropped_img = img[box[1]:box[3], box[0]:box[2]]

        # grayscale the cropped_img
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        
        # Use pytesseract to extract text from the cropped image
        text = pytesseract.image_to_string(threshold_image, config='--psm 3', lang='eng')
        # print(text)
        text_bbox.append((text, box))
    
    # Clean the extracted texts
    clean_texts_bbox = []
    for text, box in text_bbox:
        clean_text = extract_value_from_ocr(text)
        # print(clean_text)
        if clean_text:
            clean_texts_bbox.append((clean_text, box))
    
    return clean_texts_bbox

def detect_text(image_path):
    # Step 1: Extract bounding boxes
    img, bboxes = extract_bboxes(image_path)
    
    # Step 2: Merge bounding boxes
    merged_bboxes = merge_bboxes(bboxes, max_bboxes=10)
    
    # Step 3: Draw and display final bounding boxes in notebook
    # draw_bboxes(img, merged_bboxes)
    
    # Step 4: Extract text from the remaining bounding boxes
    texts_bboxs_final = extract_text_from_bboxes(img, merged_bboxes)

    # Box objects
    bounding_boxes = []
    for text, box in texts_bboxs_final:
        x1, y1, x2, y2 = box
        bbox = BoundingBox(x1, y1, x2, y2, text)
        bounding_boxes.append(bbox)
    
    return bounding_boxes
    