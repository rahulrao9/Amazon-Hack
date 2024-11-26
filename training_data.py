import re
import difflib
from typing import List, Dict
from collections import Counter
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
import cv2
import re
import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Tuple, Dict
import math
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

data = pd.read_csv("student_resource 3\\dataset\\train.csv")

# Dictionary to convert units to inches
unit_conversion = {
    "centimetre": 0.393701,
    "millimetre": 0.0393701,
    "metre": 39.3701,
    "inch": 1,
    "foot": 12,
    "yard": 36
}

# Filter rows where entity_name is height, depth, or width
filtered_data = data[data['entity_name'].isin(['height', 'depth', 'width'])]

# Convert all entity values to inches
def convert_to_inches(value, unit):
    return value * unit_conversion[unit]

# Assuming entity_value contains numeric part and a unit (e.g., '30 inch')
# Splitting and processing the entity_value into numeric and unit
filtered_data['value_in_inches'] = filtered_data['entity_value'].apply(
    lambda x: float(x.split()[0]) * unit_conversion[x.split()[1].lower()]
)

# Grouping by group_id and entity_name and calculating mean and std deviation
grouped = filtered_data.groupby(['group_id', 'entity_name'])['value_in_inches']

# Creating dictionaries for mean and standard deviation per group_id and entity_name
mean_dict = grouped.mean().unstack().to_dict()
std_dict = grouped.std().unstack().to_dict()


class BoundingBox:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, text: str):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.text = text
        self.value = None
        self.unit = None
    
    def get_centroid(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
def correct_ocr_errors(text: str) -> str:
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

    
def detect_text(image_path: str) -> List[BoundingBox]:
    # Load image with OpenCV
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to binarize the image (make it black and white)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # # Save the preprocessed image (optional, for checking)
    # cv2.imwrite('preprocessed_image.png', thresh)
    custom_config = r'--oem 3 --psm 3'
    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT, config=custom_config)
    
    boxes = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 40:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            text = data['text'][i]
            if text.strip():
                boxes.append(BoundingBox(x, y, x+w, y+h, text))
    
    return boxes

def parse_measurement(text: str) -> Tuple[float, str]:
    # Implement regex-based parser to extract numerical value and unit
    pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z"]+)'
    match = re.search(pattern, text)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        if unit in unit_conversion.keys():
            # print("YES")
            return value, unit

    return None, None

def preprocess_boxes(boxes: List[BoundingBox]) -> List[BoundingBox]:
    for box in boxes:
        box.text = correct_ocr_errors(box.text)
        box.value, box.unit = parse_measurement(box.text)
        # if box.unit is None and box.value is not None:
        #     box.unit = deduce_uom(boxes,box)
    return [box for box in boxes if box.value is not None]

# EfficientNet-based image classifier returning embeddings
class SingleBoxClassifier(nn.Module):
    def __init__(self):
        super(SingleBoxClassifier, self).__init__()
        self.efficientnet = models.efficientnet_b3(pretrained=True)
        self.efficientnet.classifier = nn.Identity()  # Remove the classifier layer of EfficientNet
        
        # EfficientNet-B3 output features
        self.feature_dim = 1536
    
    def forward(self, x):
        e = self.efficientnet(x)  # Extract features from EfficientNet
        # # Convert to NumPy matrix
        # e_matrix = e.detach().cpu().numpy()  # Detach from computation graph and move to CPU if needed

        # return e_matrix  # Return as a NumPy array
        e = e.detach().numpy()
        return e  # Return embeddings
    

def generate_input_channels(image: np.array, xi: float, yi: float) -> np.array:
    h, w = image.shape[:2]
    xi, yi = xi / w, yi / h  # Normalize coordinates

    # Convert to grayscale
    f1 = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    f1 = np.expand_dims(f1, axis=-1)  # Add channel dimension

    # Normalize coordinates
    x_coords, y_coords = np.meshgrid(np.arange(w) / w, np.arange(h) / h)
    f2 = np.abs(x_coords - xi)
    f3 = np.abs(y_coords - yi)
    
    # Stack to create 3 channels
    return np.concatenate((f1, f2[..., np.newaxis], f3[..., np.newaxis]), axis=-1)

def calculate_z_bar(value: float, unit: str, category: str) -> np.array:
    value = convert_to_inches(value, unit)
    return_list = []
    search_list = ['depth', 'width', 'height']
    for dim in search_list:
        z_alpha = (math.log(value) - mean_dict[dim][category])/std_dict[dim][category]
        if math.isnan(z_alpha):
            # print("true")
            z_alpha = math.inf
        print("z_alpha: ",z_alpha)
        z_alpha_bar = math.exp(-abs(z_alpha))
        return_list.append(z_alpha_bar)
    
    return_list.append(1.0)
    print(return_list)
    return np.array(return_list)

def process_image(image_path: str, category: str): # -> List[Dict]
    image = np.array(Image.open(image_path))
    boxes = detect_text(image_path)
    # print(boxes)
    boxes = preprocess_boxes(boxes)
    
    classifier = SingleBoxClassifier()

    print("texts detected:",len(boxes))
    
    results = []
    for box in boxes:
        centroid = box.get_centroid()
        input_channels = generate_input_channels(image, centroid[0], centroid[1])
        z_bar = calculate_z_bar(box.value, box.unit, category)
        
        # Convert input to tensor
        x_tensor = torch.tensor(input_channels).permute(2, 0, 1).unsqueeze(0).float()
        
        # Forward pass through the classifier
        with torch.no_grad():
            output = classifier(x_tensor)
        
        print(f"dims of {box} embeddings:", output.shape)
        print(f"{box} z_bar" ,z_bar)


if __name__ == "__main__":
    image = "61Drr5Mq3nL.jpg"
    category = 603688
    process_image(image_path=image, category= category)
