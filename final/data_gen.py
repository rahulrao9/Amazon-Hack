import re
import difflib
from typing import List, Dict
from collections import Counter
import numpy as np
import pandas as pd
import pytesseract
from pytesseract import Output
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
import os
import requests
from io import BytesIO
from datasets import Dataset
from final_ocr import detect_text
warnings.filterwarnings("ignore")

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

data = pd.read_csv("train.csv")

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

        # return e_matrix  # Return a tensor
        e = e.detach()
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



def process_image(image_link, category):

    response = requests.get(image_link)
    img = Image.open(BytesIO(response.content))

    image = np.array(img)
    boxes = detect_text(img)
    print("no of boxes:", len(boxes))
    classifier = SingleBoxClassifier()

    embeddings_list =[]
    text_list = []
    z_bar_list = []

    for box in boxes:

        text = box.text
        text_list.append(text)

        centroid = box.get_centroid()
        input_channels = generate_input_channels(image, centroid[0], centroid[1])
        z_bar = calculate_z_bar(box.value, box.unit, category)
        z_bar_list.append(z_bar)
        
        # Convert input to tensor
        x_tensor = torch.tensor(input_channels).permute(2, 0, 1).unsqueeze(0).float()
        
        # Forward pass through the classifier
        with torch.no_grad():
            embeddings = classifier(x_tensor)

        embeddings_list.append(embeddings)

    return embeddings_list, text_list, z_bar_list


def processor_callback(sample):
    embeddings_list, text_list, z_bar_list = process_image(sample["image_link"], sample["group_id"])   

    sample["embeddings_list"] = embeddings_list
    sample["text_list"] = text_list
    sample["z_bar_list"] = z_bar_list

    return sample
    

if __name__ == "__main__":
    df = pd.read_csv("pivot_train.csv")
    df = df[:10]
    H_DF = Dataset.from_pandas(df)

    H_DF = H_DF.map(processor_callback, num_proc = 4)

    H_DF = H_DF.to_pandas()

    H_DF.to_csv("stage1_train.csv", index = False)